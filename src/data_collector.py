from __future__ import annotations
"""
data_collector.py
-----------------
Data source: LIAR Dataset (Wang, ACL 2017)
  - 12,836 real statements from PolitiFact, labelled by human fact-checkers
  - Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
  - Place train.tsv, test.tsv, valid.tsv in the data/liar/ folder

Loading priority:
  1. LIAR TSV files  ->  real claims, synthetic propagation edges
  2. Cached CSVs     ->  previously processed data (fast reload)
  3. Synthetic       ->  fully generated fallback (no files needed)

Why synthetic propagation?
  Twitter's API became paywalled in 2023 (~$100/month). Real diffusion graphs
  for these claims are no longer freely accessible. The propagation layer is
  calibrated to match empirically observed patterns from:
  Vosoughi, Roy & Aral, "The spread of true and false news online",
  Science, 2018 (DOI: 10.1126/science.aap9559)
"""

import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── constants ─────────────────────────────────────────────────────────────────
PLATFORMS   = ["Twitter", "Facebook", "Reddit", "Telegram", "WhatsApp"]
COMMUNITIES = ["Health", "Politics", "Science", "Conspiracy", "Mainstream"]

# LIAR has 6 labels -- map to binary fake / real
# "pants-fire" and "false" are clearly fake
# "barely-true" is treated as fake (misleading even if not outright false)
# "half-true", "mostly-true", "true" treated as real
LIAR_LABEL_MAP = {
    "pants-fire":   "fake",
    "false":        "fake",
    "barely-true":  "fake",
    "mostly-false": "fake",
    "pants_fire":   "fake",
    "half-true":    "real",
    "mostly-true":  "real",
    "true":         "real",
}

# LIAR 14-column TSV schema (no header row in the file)
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count",
    "venue",
]


# =============================================================================
# 1.  LIAR LOADER
# =============================================================================

def load_liar_tsv(data_dir: str = "data") -> pd.DataFrame | None:
    """
    Loads train.tsv + test.tsv + valid.tsv from data/liar/.
    Returns a combined DataFrame or None if files are not found.

    Download instructions
    --------------------
    1. Go to: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    2. Unzip -- you get train.tsv, test.tsv, valid.tsv
    3. Create folder:  <project>/data/liar/
    4. Move the three TSV files into it
    5. Re-run the app -- this function picks them up automatically
    """
    liar_dir  = Path(data_dir) / "liar"
    tsv_files = ["train.tsv", "test.tsv", "valid.tsv"]
    found     = [liar_dir / f for f in tsv_files if (liar_dir / f).exists()]

    if not found:
        return None

    frames = []
    for path in found:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=LIAR_COLUMNS,
                dtype=str,
                on_bad_lines="skip",
            )
            frames.append(df)
            print(f"  📄  Loaded {path.name}  ({len(df):,} rows)")
        except Exception as exc:
            print(f"  ⚠️   Could not read {path.name}: {exc}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    print(f"  ✅  Combined LIAR: {len(combined):,} statements")
    return combined


def parse_liar(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw LIAR DataFrame into the claims schema used by the project.

    Output columns
    --------------
    claim_id, text, label (fake/real), speaker, subject, party, venue,
    origin_platform, origin_community, origin_time,
    virality_score, credibility_history
    """
    df = raw_df.copy()

    # binary label
    df["label"] = (
        df["label"].str.lower().str.strip().map(LIAR_LABEL_MAP)
    )
    df = df.dropna(subset=["label", "statement"])
    df["statement"] = df["statement"].str.strip()
    df = df[df["statement"].str.len() > 10]

    # claim_id
    df["claim_id"] = (
        df["id"].str.replace(".json", "", regex=False).str.strip()
    )

    # credibility history
    count_cols = [
        "barely_true_count", "false_count", "half_true_count",
        "mostly_true_count", "pants_fire_count",
    ]
    for col in count_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    false_cols = ["barely_true_count", "false_count", "pants_fire_count"]
    df["false_history"] = df[false_cols].sum(axis=1)
    df["total_history"] = df[count_cols].sum(axis=1)
    df["credibility_history"] = (
        1 - df["false_history"] / df["total_history"].replace(0, 1)
    ).round(3)

    # virality score calibrated to Vosoughi et al. 2018
    rng      = np.random.default_rng(SEED)
    n        = len(df)
    fake_mask = df["label"].values == "fake"
    virality = np.where(
        fake_mask,
        rng.beta(a=3, b=2, size=n),
        rng.beta(a=2, b=4, size=n),
    )
    df["virality_score"] = np.round(virality, 3)

    # synthetic origin metadata
    rng2 = random.Random(SEED)
    df["origin_platform"]  = [rng2.choice(PLATFORMS)   for _ in range(n)]
    df["origin_community"] = [rng2.choice(COMMUNITIES) for _ in range(n)]
    base = datetime(2024, 1, 1)
    df["origin_time"] = [
        base + timedelta(days=rng2.randint(0, 364)) for _ in range(n)
    ]

    return df[[
        "claim_id", "statement", "label",
        "speaker", "subject", "party", "venue",
        "origin_platform", "origin_community", "origin_time",
        "virality_score", "credibility_history",
    ]].rename(columns={"statement": "text"}
    ).drop_duplicates(subset=["claim_id"]
    ).reset_index(drop=True)


# =============================================================================
# 2.  PROPAGATION EDGE GENERATOR
# =============================================================================

def _uid(seed_str: str) -> str:
    return hashlib.md5(seed_str.encode()).hexdigest()[:10]


def generate_propagation_edges(
    claims_df: pd.DataFrame,
    max_shares_per_claim: int = 200,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """
    Generates synthetic propagation edges for each claim.

    Calibration (Vosoughi et al., Science 2018)
    -------------------------------------------
    - False news reached more people faster and more broadly than true news
    - False news was 70% more likely to be retweeted
    - True news took ~6x longer to reach 1,500 people
    - Fake cascades were deeper and broader
    """
    rng_r = random.Random(SEED)
    rng_n = np.random.default_rng(SEED)

    working = claims_df.copy()
    if sample_n and sample_n < len(working):
        working = working.sample(n=sample_n, random_state=SEED).reset_index(drop=True)

    edges = []
    for _, row in working.iterrows():
        claim_id    = row["claim_id"]
        label       = row["label"]
        origin_time = pd.to_datetime(row["origin_time"])

        if label == "fake":
            n_shares     = rng_r.randint(60, max_shares_per_claim)
            speed_factor = 0.35
            sent_centre  = 0.35
            sent_std     = 0.25
        else:
            n_shares     = rng_r.randint(10, max(12, max_shares_per_claim // 4))
            speed_factor = 2.2
            sent_centre  = 0.60
            sent_std     = 0.15

        prev_user    = f"user_{_uid(claim_id + 'origin')}"
        current_time = origin_time

        for share_idx in range(n_shares):
            current_time += timedelta(
                hours=float(rng_n.exponential(scale=speed_factor))
            )
            new_user  = f"user_{_uid(claim_id + str(share_idx))}"
            depth     = min(share_idx // 8 + 1, 12)
            sentiment = float(np.clip(
                rng_n.normal(loc=sent_centre, scale=sent_std), 0.0, 1.0
            ))

            edges.append({
                "claim_id":       claim_id,
                "source_user":    prev_user,
                "target_user":    new_user,
                "platform":       rng_r.choice(PLATFORMS),
                "community":      rng_r.choice(COMMUNITIES),
                "timestamp":      current_time,
                "depth":          depth,
                "sentiment_score": round(sentiment, 3),
                "is_debunk":      rng_r.random() < (0.05 if label == "fake" else 0.02),
            })

            if rng_r.random() < 0.3 and share_idx > 5:
                prev_user = f"user_{_uid(claim_id + str(rng_r.randint(0, share_idx - 1)))}"
            else:
                prev_user = new_user

    return pd.DataFrame(edges)


# =============================================================================
# 3.  PURE SYNTHETIC FALLBACK
# =============================================================================

_FAKE_CLAIMS = [
    "5G towers are spreading the virus through radio waves",
    "Drinking bleach cures respiratory infections",
    "The government is hiding a cure found in common herbs",
    "New study proves vaccines cause autism in children",
    "Scientists confirm moon landing was staged in Hollywood",
    "Microchips are being inserted via routine flu shots",
    "Eating certain foods reverses all chronic disease",
    "Secret chemtrails are altering human DNA globally",
]

_TRUE_CLAIMS = [
    "Handwashing for 20 seconds reduces infection spread significantly",
    "New mRNA vaccine platform shows promise against multiple diseases",
    "Scientists discover exoplanet with potential water signatures",
    "Renewable energy surpasses coal in global electricity generation",
    "Study confirms Mediterranean diet lowers cardiovascular risk",
    "WHO approves new malaria vaccine for children under five",
    "CERN researchers detect rare particle decay event",
    "Ocean cleanup project removes 100 tonnes of plastic annually",
]


def generate_synthetic_dataset(
    n_claims: int = 16,
    max_shares_per_claim: int = 300,
    output_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure synthetic fallback -- runs with zero external files."""
    Path(output_dir).mkdir(exist_ok=True)
    rng = random.Random(SEED)

    all_claims = (
        [(c, "fake") for c in _FAKE_CLAIMS]
        + [(c, "real") for c in _TRUE_CLAIMS]
    )
    selected = rng.sample(all_claims, min(n_claims, len(all_claims)))

    claims = []
    for idx, (text, label) in enumerate(selected):
        n_shares = (
            rng.randint(80, max_shares_per_claim)
            if label == "fake"
            else rng.randint(20, max_shares_per_claim // 3)
        )
        claims.append({
            "claim_id":            f"synth_{idx:03d}",
            "text":                text,
            "label":               label,
            "speaker":             "unknown",
            "subject":             "general",
            "party":               "none",
            "venue":               "social media",
            "origin_platform":     rng.choice(PLATFORMS),
            "origin_community":    rng.choice(COMMUNITIES),
            "origin_time":         datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 180)),
            "virality_score":      round(n_shares / max_shares_per_claim, 3),
            "credibility_history": 0.5,
        })

    claims_df = pd.DataFrame(claims)
    spread_df = generate_propagation_edges(
        claims_df, max_shares_per_claim=max_shares_per_claim
    )

    claims_df.to_csv(f"{output_dir}/claims.csv", index=False)
    spread_df.to_csv(f"{output_dir}/spread_edges.csv", index=False)
    print(f"✅  [Synthetic] {len(claims_df)} claims | {len(spread_df):,} edges")
    return claims_df, spread_df


# =============================================================================
# 4.  MAIN LOADER
# =============================================================================

def load_dataset(
    data_dir: str = "data",
    max_claims: int = 200,
    max_shares_per_claim: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    3-tier loading priority:

    Tier 1 -- LIAR TSV files (real PolitiFact data)
        Place train.tsv / test.tsv / valid.tsv in data/liar/
        Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

    Tier 2 -- Cached CSVs (fast reload after first parse)
        data/claims.csv + data/spread_edges.csv

    Tier 3 -- Fully synthetic fallback (zero files needed)

    Parameters
    ----------
    max_claims           : cap loaded claims (laptop safety, default 200)
    max_shares_per_claim : cap propagation edges per claim (default 200)
    """
    Path(data_dir).mkdir(exist_ok=True)
    claims_path = Path(data_dir) / "claims.csv"
    edges_path  = Path(data_dir) / "spread_edges.csv"

    # Tier 1 -- LIAR
    raw_liar = load_liar_tsv(data_dir)
    if raw_liar is not None:
        print("🗂️   Parsing LIAR dataset …")
        claims_df = parse_liar(raw_liar)

        if len(claims_df) > max_claims:
            half = max_claims // 2
            fake_s = claims_df[claims_df["label"] == "fake"].sample(
                n=min(half, (claims_df["label"] == "fake").sum()),
                random_state=SEED
            )
            real_s = claims_df[claims_df["label"] == "real"].sample(
                n=min(half, (claims_df["label"] == "real").sum()),
                random_state=SEED
            )
            claims_df = pd.concat([fake_s, real_s]).reset_index(drop=True)
            print(f"  ✂️   Sampled {len(claims_df)} claims (laptop-safe cap)")

        print(f"  🔗  Generating propagation edges …")
        spread_df = generate_propagation_edges(
            claims_df, max_shares_per_claim=max_shares_per_claim
        )

        claims_df.to_csv(claims_path, index=False)
        spread_df.to_csv(edges_path,  index=False)
        print(f"  💾  Cached to {data_dir}/")
        return _clean(claims_df, spread_df)

    # Tier 2 -- cache
    if claims_path.exists() and edges_path.exists():
        print("📂  Loading cached dataset …")
        claims_df = pd.read_csv(claims_path, parse_dates=["origin_time"])
        spread_df = pd.read_csv(edges_path,  parse_dates=["timestamp"])
        return _clean(claims_df, spread_df)

    # Tier 3 -- synthetic
    print("🔧  LIAR files not found — using synthetic data.")
    print("    To use real data, download liar_dataset.zip and unzip TSVs to data/liar/")
    print("    URL: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
    return generate_synthetic_dataset(
        output_dir=data_dir,
        max_shares_per_claim=max_shares_per_claim,
    )


# =============================================================================
# 5.  CLEANING
# =============================================================================

def _clean(
    claims_df: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise types, drop nulls, compute hours_since_origin."""

    claims_df["label"] = claims_df["label"].str.lower().str.strip()
    claims_df["text"]  = claims_df["text"].astype(str).str.strip()
    claims_df = claims_df.dropna(subset=["claim_id", "text", "label"])
    claims_df = claims_df[claims_df["label"].isin(["fake", "real"])]

    for col, default in [
        ("speaker", "unknown"), ("subject", "general"),
        ("party", "none"), ("venue", "unknown"),
        ("credibility_history", 0.5),
    ]:
        if col not in claims_df.columns:
            claims_df[col] = default

    spread_df = spread_df.dropna(subset=["claim_id", "source_user", "target_user"])
    spread_df["timestamp"] = pd.to_datetime(spread_df["timestamp"], errors="coerce")
    spread_df = spread_df.dropna(subset=["timestamp"])
    spread_df["hours_since_origin"] = (
        spread_df.groupby("claim_id")["timestamp"]
        .transform(lambda s: (s - s.min()).dt.total_seconds() / 3600)
    )
    spread_df["is_debunk"] = spread_df["is_debunk"].fillna(False).astype(bool)

    n_fake = (claims_df["label"] == "fake").sum()
    n_real = (claims_df["label"] == "real").sum()
    print(
        f"✅  Dataset ready: {len(claims_df)} claims "
        f"({n_fake} fake / {n_real} real) | "
        f"{len(spread_df):,} propagation edges"
    )
    return claims_df, spread_df


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    c, e = load_dataset()
    print("\n── Claims sample ──")
    print(c[["claim_id", "label", "speaker", "virality_score"]].head(8))
    print("\n── Edges sample ──")
    print(e[["claim_id", "source_user", "depth", "hours_since_origin"]].head(6))
    print(f"\nLabel balance:\n{c['label'].value_counts()}")
