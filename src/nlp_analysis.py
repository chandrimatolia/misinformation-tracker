from __future__ import annotations
"""
nlp_analysis.py
---------------
NLP layer for the Misinformation Spread Tracker:
  1. Zero-shot claim classification  (real vs. fake signals)
  2. Sentiment analysis across the propagation chain
  3. Claim mutation tracking        (how wording changes as it spreads)
  4. Keyword / topic extraction
  5. Virality risk scoring          (heuristic + text features)

Uses only lightweight, offline-friendly libraries so the project runs
without GPU or API keys.  Swap the classifier for a HuggingFace pipeline
(e.g. 'facebook/bart-large-mnli') for production-grade results.
"""

import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEXT PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
    "that", "this", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "not", "no", "for",
    "on", "at", "by", "with", "from", "as", "but", "if", "they",
    "their", "them", "we", "our", "you", "your", "he", "she", "his",
    "her", "its", "my", "i", "up", "so", "all", "more", "than",
}

# Sensationalist signal words common in misinformation
_ALARM_WORDS = {
    # Conspiracy & suppression
    "secret", "suppressed", "coverup", "conspiracy", "hidden", "agenda",
    "exposed", "hoax", "truth", "wake", "deep", "they", "whistleblower",
    "leaked", "classified", "declassified", "blackout", "censored",
    "silenced", "banned", "forbidden", "confiscated", "scrubbed",
    # Sensationalism
    "shocking", "bombshell", "explosive", "stunning", "unbelievable",
    "incredible", "outrageous", "scandalous", "disturbing", "alarming",
    "breaking", "urgent", "alert", "warning", "danger", "crisis",
    "emergency", "catastrophe", "apocalypse", "collapse", "meltdown",
    # Health misinformation
    "cure", "miracle", "remedy", "detox", "toxin", "poison", "deadly",
    "toxic", "carcinogenic", "contaminated", "tainted", "laced",
    "microchip", "nanoparticle", "bioweapon", "depopulation",
    # False certainty
    "proven", "guaranteed", "100%", "forever", "instant", "overnight",
    "always", "never", "absolute", "definitive", "undeniable", "irrefutable",
    # Manipulation & deception
    "lies", "liars", "fake", "fraud", "scam", "rigged", "stolen",
    "manipulated", "fabricated", "planted", "staged", "scripted",
    "psyop", "propaganda", "brainwashed", "indoctrinated",
    # Big pharma / establishment
    "bigpharma", "elites", "globalists", "cabal", "regime", "tyrants",
    "puppet", "controlled", "owned", "bought", "paid",
    # Urgency & fear
    "must", "immediately", "now", "before", "destroyed", "deleted",
    "share", "spread", "everyone", "nobody", "nobody",
}

# Credibility signal words common in factual content
_CREDIBILITY_WORDS = {
    # Academic & scientific
    "study", "research", "published", "journal", "peer-reviewed",
    "scientists", "researchers", "data", "evidence", "according",
    "university", "institute", "analysis", "results", "findings",
    "confirmed", "measured", "observed", "reported", "verified",
    # Statistical & factual
    "percent", "statistics", "survey", "sample", "methodology",
    "controlled", "randomised", "meta-analysis", "systematic",
    "clinical", "trial", "experiment", "hypothesis", "peer",
    # Attribution & sourcing
    "cited", "referenced", "documented", "recorded", "official",
    "government", "agency", "department", "committee", "commission",
    "spokesperson", "statement", "announcement", "report", "audit",
    # Hedging & nuance
    "suggests", "indicates", "appears", "estimated", "approximately",
    "roughly", "likely", "possible", "uncertain", "preliminary",
}


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return [w for w in preprocess(text).split() if w not in _STOPWORDS]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HEURISTIC CREDIBILITY SCORER
#     (Replace with HuggingFace pipeline for production)
# ─────────────────────────────────────────────────────────────────────────────

def credibility_score(text: str) -> dict:
    """
    Returns a score in [0, 1] where 0 = highly suspicious, 1 = credible.
    Also returns contributing signals.
    """
    tokens = set(tokenize(text))
    alarm_hits      = tokens & _ALARM_WORDS
    credibility_hits = tokens & _CREDIBILITY_WORDS

    # All-caps ratio
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Exclamation / question mark density
    punct_density = (text.count("!") + text.count("?")) / max(len(text.split()), 1)

    # Numeric evidence (numbers suggest data-backed claims)
    has_numbers = bool(re.search(r"\d+\.?\d*\s*(%|mg|kg|km|ml|°|percent)", text.lower()))

    alarm_score  = len(alarm_hits) / max(len(tokens), 1)
    cred_score   = len(credibility_hits) / max(len(tokens), 1)

    raw = (
        0.50 * (1 - alarm_score)
        + 0.25 * cred_score
        + 0.10 * (1 - caps_ratio * 5)
        + 0.10 * (1 - punct_density * 3)
        + 0.05 * int(has_numbers)
    )
    score = float(np.clip(raw, 0.0, 1.0))

    return {
        "credibility_score": round(score, 3),
        "label_prediction":  "likely_real" if score >= 0.45 else "likely_fake",
        "alarm_words":       list(alarm_hits),
        "credibility_words": list(credibility_hits),
        "caps_ratio":        round(caps_ratio, 3),
        "punct_density":     round(punct_density, 3),
        "has_numeric_evidence": has_numbers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SENTIMENT ANALYSIS  (lightweight lexicon-based)
# ─────────────────────────────────────────────────────────────────────────────

_POSITIVE = {
    "good","great","safe","effective","beneficial","healthy","proven",
    "success","positive","improve","cure","help","better","excellent",
    "wonderful","amazing","hope","promise","advance","discovery",
}
_NEGATIVE = {
    "bad","dangerous","harmful","toxic","deadly","kill","death","risk",
    "fear","panic","crisis","scary","threat","damage","injury","evil",
    "corrupt","lie","fake","hoax","cover","suppress","ban","hide",
}


def sentiment_score(text: str) -> float:
    """Returns sentiment in [-1, 1] where -1 is very negative."""
    tokens = tokenize(text)
    pos = sum(1 for t in tokens if t in _POSITIVE)
    neg = sum(1 for t in tokens if t in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def sentiment_over_time(
    claim_id: str,
    spread_df: pd.DataFrame,
    sample_texts: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns sentiment evolution binned by depth level.
    If sample_texts is provided (one per share), uses those; otherwise
    uses the stored sentiment_score column with small synthetic drift.
    """
    edges = spread_df[spread_df["claim_id"] == claim_id].copy()
    if edges.empty:
        return pd.DataFrame()

    if "sentiment_score" in edges.columns:
        # Add mild cumulative drift to simulate mutation
        edges = edges.sort_values("hours_since_origin")
        edges["rolling_sentiment"] = (
            edges["sentiment_score"]
            .rolling(window=5, min_periods=1)
            .mean()
            .round(3)
        )
    else:
        edges["rolling_sentiment"] = 0.5

    return (
        edges.groupby("depth")["rolling_sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"rolling_sentiment": "avg_sentiment"})
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLAIM MUTATION TRACKER
# ─────────────────────────────────────────────────────────────────────────────

# Simulated claim mutations (in a real project these come from tweet texts)
_MUTATION_TEMPLATES = {
    "5G towers are spreading the virus through radio waves": [
        "5G towers spread disease via invisible waves",
        "New research links 5G signals to illness outbreaks",
        "5G radiation proven to weaken your immune system",
        "Experts: 5G networks responsible for mystery sickness",
        "BANNED: the truth about 5G and public health risks",
    ],
    "Drinking bleach cures respiratory infections": [
        "Household chemical found to fight lung infections",
        "Natural remedy using common cleaners stops infections",
        "Doctors don't want you to know this household cure",
        "Oxidizing agents shown to reduce infection symptoms",
    ],
}


def get_mutation_similarity(claim_text: str) -> pd.DataFrame:
    """
    Shows how semantic similarity decreases as a claim mutates.
    Returns a DataFrame with version, text snippet, and similarity to original.
    """
    mutations = _MUTATION_TEMPLATES.get(
        claim_text,
        [
            claim_text + " (shared version 1)",
            claim_text.replace("are", "were") + " — LEAKED",
            "BREAKING: " + claim_text[:40] + "…",
            claim_text.upper()[:50],
        ],
    )

    all_texts = [claim_text] + mutations
    vec = TfidfVectorizer().fit_transform(all_texts)
    sims = cosine_similarity(vec[0:1], vec[1:]).flatten()

    rows = []
    for i, (mut, sim) in enumerate(zip(mutations, sims)):
        rows.append({
            "version":    f"v{i+1}",
            "text":       mut[:70] + ("…" if len(mut) > 70 else ""),
            "similarity": round(float(sim), 3),
            "drift":      round(1 - float(sim), 3),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  KEYWORD / TOPIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_keywords(texts: list[str], top_n: int = 15) -> list[tuple[str, float]]:
    """TF-IDF keyword extraction across a list of texts."""
    if not texts:
        return []
    vec = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    try:
        X = vec.fit_transform(texts)
        scores = X.mean(axis=0).A1
        vocab  = vec.get_feature_names_out()
        ranked = sorted(zip(vocab, scores), key=lambda x: -x[1])
        return [(w, round(float(s), 4)) for w, s in ranked[:top_n]]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VIRALITY RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────

def virality_risk_score(
    claim_text: str,
    graph_metrics: dict | None = None,
) -> dict:
    """
    Combines NLP signals + graph metrics into a 0-100 virality risk score.
    Higher = more likely to go viral.
    """
    cred   = credibility_score(claim_text)
    sent   = sentiment_score(claim_text)

    nlp_risk = (
        0.40 * (1 - cred["credibility_score"])   # low credibility → high risk
        + 0.20 * abs(sent)                        # emotional content spreads
        + 0.20 * min(len(cred["alarm_words"]) / 5, 1.0)
        + 0.20 * cred["caps_ratio"] * 3
    )

    graph_risk = 0.0
    if graph_metrics:
        v  = graph_metrics.get("virality_score", 0)
        d  = min(graph_metrics.get("max_depth", 0) / 10, 1)
        b  = min(graph_metrics.get("max_breadth", 0) / 50, 1)
        graph_risk = (v + d + b) / 3

    weight_nlp   = 0.5 if graph_metrics else 1.0
    weight_graph = 0.5 if graph_metrics else 0.0
    combined = weight_nlp * nlp_risk + weight_graph * graph_risk
    score_100 = round(float(np.clip(combined * 100, 0, 100)), 1)

    return {
        "virality_risk": score_100,
        "risk_level":    "🔴 High" if score_100 >= 65 else ("🟡 Medium" if score_100 >= 35 else "🟢 Low"),
        "nlp_risk":      round(nlp_risk * 100, 1),
        "graph_risk":    round(graph_risk * 100, 1),
        "top_signals":   cred["alarm_words"][:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_all_claims(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Adds NLP columns to the claims dataframe."""
    records = []
    for _, row in claims_df.iterrows():
        cred = credibility_score(row["text"])
        sent = sentiment_score(row["text"])
        risk = virality_risk_score(row["text"])
        records.append({
            "claim_id":          row["claim_id"],
            "credibility_score": cred["credibility_score"],
            "nlp_label":         cred["label_prediction"],
            "sentiment":         sent,
            "virality_risk":     risk["virality_risk"],
            "risk_level":        risk["risk_level"],
            "alarm_words":          ", ".join(cred["alarm_words"]),
            "alarm_word_count":     len(cred["alarm_words"]),
            "credibility_word_count": len(cred["credibility_words"]),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_collector import load_dataset

    claims_df, spread_df = load_dataset()
    nlp_df = analyse_all_claims(claims_df)
    print(nlp_df[["claim_id", "credibility_score", "nlp_label", "virality_risk", "risk_level"]])

    print("\n── Mutation tracker ──")
    sample = claims_df["text"].iloc[0]
    print(get_mutation_similarity(sample))
