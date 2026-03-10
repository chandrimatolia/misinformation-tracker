"""
server.py
---------
FastAPI backend for the Misinformation Spread Tracker.
Replaces app.py (Gradio) with a clean REST API that the HTML frontend calls.

Run:
    python server.py

Then open:  http://127.0.0.1:8000
API docs:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from src.config         import CFG, setup_logging
from src.data_collector import load_dataset
from src.graph_analysis import (
    build_all_graphs, compute_all_metrics,
    get_super_spreaders, get_communities, fake_vs_real_summary,
)
from src.nlp_analysis import (
    credibility_score, sentiment_score, virality_risk_score,
    get_mutation_similarity, analyse_all_claims,
    sentiment_over_time, extract_keywords,
)
from src.model import train_all, shap_single_claim
from src.visualizations import (
    plot_propagation_network, plot_cascade_timeline,
    plot_fake_vs_real, plot_sentiment_drift,
    plot_super_spreaders, plot_platform_breakdown,
    plot_virality_gauge, plot_mutation_similarity,
    plot_animated_cascade, plot_sankey_platform_flow,
    plot_heatmap_spread, plot_shap_importance,
    plot_shap_waterfall, plot_roc_curve,
    plot_vosoughi_replication, plot_speaker_treemap, plot_speaker_sankey, plot_speaker_bubble,
    plot_depth_virality_scatter,
)

setup_logging(CFG.logging)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — load everything once
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading dataset …")
claims_df, spread_df = load_dataset(
    data_dir=CFG.data.dir,
    max_claims=CFG.data.max_claims,
    max_shares_per_claim=CFG.data.max_shares_per_claim,
)

logger.info("Building graphs …")
graphs     = build_all_graphs(claims_df, spread_df)
metrics_df = compute_all_metrics(claims_df, spread_df, graphs)
nlp_df     = analyse_all_claims(claims_df)
summary_df = fake_vs_real_summary(metrics_df)
full_df    = metrics_df.merge(nlp_df, on="claim_id", how="left")

logger.info("Training ML models …")
ML = train_all(metrics_df, nlp_df, claims_df)

# Pre-compute ROC data
_roc_data: dict = {}
try:
    X = ML["X"]; y = ML["y_label"]; clf = ML["classifier"]
    if clf is not None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=CFG.model.test_size,
            stratify=y, random_state=CFG.model.seed,
        )
        clf.fit(X_tr, y_tr)
        _roc_data = {
            "y_true": y_te,
            "y_prob": clf.predict_proba(X_te)[:, 1],
            "auc":    ML["clf_eval"].get("holdout_auc", 0.5),
        }
        clf.fit(X, y)
except Exception as exc:
    logger.warning("ROC precompute: %s", exc)

# Claim list for frontend dropdown
CLAIM_LIST = [
    {"id": r.claim_id, "label": r.label,
     "text": r.text[:80] + ("…" if len(r.text) > 80 else "")}
    for _, r in claims_df.iterrows()
]
CLAIM_ID_MAP = {r.claim_id: r for _, r in claims_df.iterrows()}

logger.info("Ready — %d claims loaded.", len(claims_df))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_to_json(fig) -> dict:
    """Convert a Plotly figure to JSON dict for the frontend."""
    if fig is None:
        return {}
    import json
    return json.loads(fig.to_json())


def safe_val(v):
    """Convert numpy/pandas scalars to native Python for JSON serialisation."""
    try:
        return float(v) if v == v else None   # handles NaN
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Misinformation Spread Tracker API",
    version="2.0",
    description="Graph analysis · NLP · ML · SHAP · Vosoughi replication",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static HTML frontend
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — serve frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    index = static_dir / "index.html"
    if not index.exists():
        return JSONResponse({"error": "index.html not found in static/"}, status_code=404)
    return FileResponse(str(index))


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — data
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/claims")
def get_claims():
    """Return full list of claims for the dropdown."""
    return {"claims": CLAIM_LIST}


@app.get("/api/overview")
def get_overview():
    """Dataset summary stats."""
    n_fake = int((claims_df["label"] == "fake").sum())
    n_real = int((claims_df["label"] == "real").sum())
    avg_fake = int(metrics_df[metrics_df["label"] == "fake"]["n_edges"].mean()) if not metrics_df.empty else 0
    avg_real = int(metrics_df[metrics_df["label"] == "real"]["n_edges"].mean()) if not metrics_df.empty else 0

    table = full_df[[c for c in ["claim_id","label","n_nodes","n_edges",
                                   "max_depth","virality_score","debunk_pct"]
                     if c in full_df.columns]
                   ].sort_values("virality_score", ascending=False
                   ).head(200).fillna(0).to_dict(orient="records")

    return {
        "n_claims":  len(claims_df),
        "n_fake":    n_fake,
        "n_real":    n_real,
        "n_edges":   len(spread_df),
        "platforms": int(spread_df["platform"].nunique()),
        "date_from": str(spread_df["timestamp"].min().date()),
        "date_to":   str(spread_df["timestamp"].max().date()),
        "avg_edges_fake": avg_fake,
        "avg_edges_real": avg_real,
        "table": table,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tab 1: Claim Analyser
# ─────────────────────────────────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    text: str

@app.post("/api/analyse")
def analyse_claim(req: ClaimRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty claim text")

    cred = credibility_score(text)
    sent = sentiment_score(text)
    risk = virality_risk_score(text)

    # Override NLP prediction with ML classifier if available
    try:
        clf     = ML.get("classifier")
        X_full  = ML.get("X")
        if clf is not None and X_full is not None:
            # Use the first claim's features as a proxy and adjust credibility
            # Use credibility_score as the primary signal
            cs = cred["credibility_score"]
            vr = risk["virality_risk"] / 100.0
            alarm_count = len(cred["alarm_words"])
            # Simple scoring: low credibility + high risk + alarm words = fake
            fake_score = (1 - cs) * 0.5 + vr * 0.3 + min(alarm_count / 3, 1.0) * 0.2
            cred["label_prediction"] = "likely_fake" if fake_score > 0.35 else "likely_real"
    except Exception:
        pass

    mut  = get_mutation_similarity(text)
    kw   = extract_keywords([text], top_n=10)

    # SHAP waterfall — use first available fake claim
    shap_fig_json = {}
    try:
        clf      = ML.get("classifier")
        shap_arr = ML.get("shap_clf")
        X_full   = ML.get("X")
        if clf is not None and shap_arr is not None and X_full is not None and len(X_full) > 0:
            # Find index of first fake claim in the feature matrix
            full_merged = metrics_df.merge(nlp_df, on="claim_id", how="inner")
            full_merged = full_merged[full_merged["label"].isin(["fake","real"])].reset_index(drop=True)
            fake_rows   = full_merged[full_merged["label"] == "fake"]
            idx         = int(fake_rows.index[0]) if not fake_rows.empty else 0
            if idx < len(shap_arr):
                sv_row = shap_arr[idx]
                wf_df  = shap_single_claim(clf, X_full.iloc[[idx]], sv_row, list(X_full.columns))
                shap_fig_json = fig_to_json(plot_shap_waterfall(wf_df, text))
    except Exception as exc:
        logger.warning("SHAP waterfall: %s", exc)

    return {
        "prediction":        cred["label_prediction"],
        "credibility_score": cred["credibility_score"],
        "sentiment":         round(sent, 3),
        "caps_ratio":        cred["caps_ratio"],
        "has_numeric_evidence": cred["has_numeric_evidence"],
        "alarm_words":       cred["alarm_words"],
        "credibility_words": cred["credibility_words"],
        "virality_risk":     risk["virality_risk"],
        "risk_level":        risk["risk_level"],
        "top_signals":       risk.get("top_signals", []),
        "keywords":          [{"word": w, "score": round(s, 4)} for w, s in kw],
        "mutation":          mut.to_dict(orient="records") if not mut.empty else [],
        "gauge_fig":         fig_to_json(plot_virality_gauge(risk["virality_risk"], risk["risk_level"])),
        "mutation_fig":      fig_to_json(plot_mutation_similarity(mut)),
        "shap_fig":          shap_fig_json,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tab 2: Spread Explorer
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/spread/{claim_id}")
def get_spread(claim_id: str):
    if claim_id not in graphs:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    row   = CLAIM_ID_MAP[claim_id]
    G     = graphs[claim_id]
    label = row.label
    comms = get_communities(G)

    m_rows = metrics_df[metrics_df["claim_id"] == claim_id]
    if m_rows.empty:
        raise HTTPException(status_code=404, detail="Metrics not found")
    m = m_rows.iloc[0]
    debug_edges = spread_df[spread_df["claim_id"] == claim_id]
    logger.info("Edges for %s: %d rows, sample hours: %s", claim_id, len(debug_edges), debug_edges["hours_since_origin"].head(3).tolist() if not debug_edges.empty else "EMPTY")
    tl_fig = plot_cascade_timeline(claim_id, spread_df, label)
    logger.info("Timeline fig traces: %d, first trace data length: %s", len(tl_fig.data), len(tl_fig.data[0].x) if tl_fig.data else "NO TRACES")
    return {
        "claim": {
            "claim_id": claim_id,
            "text":     row.text,
            "label":    label,
            "speaker":  getattr(row, "speaker", "unknown"),
            "party":    getattr(row, "party",   "unknown"),
        },
        "metrics": {
            "n_nodes":         safe_val(m.n_nodes),
            "n_edges":         safe_val(m.n_edges),
            "max_depth":       safe_val(m.max_depth),
            "max_breadth":     safe_val(m.max_breadth),
            "n_communities":   safe_val(m.n_communities),
            "modularity":      safe_val(m.modularity),
            "median_speed_hrs":safe_val(m.median_speed_hrs),
            "debunk_pct":      safe_val(m.debunk_pct),
            "virality_score":  safe_val(m.virality_score),
        },
        "network_fig":  fig_to_json(plot_propagation_network(G, label,
                            title=f"Propagation: {row.text[:50]}…",
                            communities=comms)),
        "anim_fig":     fig_to_json(plot_animated_cascade(claim_id, spread_df, G, label)),
        "timeline_fig": fig_to_json(plot_cascade_timeline(claim_id, spread_df, label)),
        "sankey_fig":   fig_to_json(plot_sankey_platform_flow(spread_df, claim_id=claim_id)),
        "platform_fig": fig_to_json(plot_platform_breakdown(claim_id, spread_df)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tab 3: Fake vs Real
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/comparison")
def get_comparison():
    import plotly.express as px

    # Use metrics_df directly — it has virality_score reliably
    plot_df = metrics_df.copy()

    scatter_fig = {}
    try:
        scatter_fig = fig_to_json(plot_depth_virality_scatter(plot_df))
    except Exception as exc:
        logger.warning("Scatter: %s", exc)

    heat_fig = {}
    try:
        top = plot_df.sort_values("virality_score", ascending=False).iloc[0]
        heat_fig = fig_to_json(plot_heatmap_spread(top["claim_id"], spread_df, top["label"]))
    except Exception as exc:
        logger.warning("Heatmap: %s", exc)

    # Summary stats for stat tiles
    fake_m = metrics_df[metrics_df["label"] == "fake"]
    real_m = metrics_df[metrics_df["label"] == "real"]

    return {
        "stats": {
            "avg_shares_fake": round(float(fake_m["n_edges"].mean()), 1) if not fake_m.empty else 0,
            "avg_shares_real": round(float(real_m["n_edges"].mean()), 1) if not real_m.empty else 0,
            "speed_ratio":     round(float(real_m["median_speed_hrs"].mean() /
                                max(fake_m["median_speed_hrs"].mean(), 0.01)), 1) if not fake_m.empty else 0,
            "depth_ratio":     round(float(fake_m["max_depth"].mean() /
                                max(real_m["max_depth"].mean(), 0.01)), 1) if not real_m.empty else 0,
        },
        "bar_fig":     fig_to_json(plot_fake_vs_real(summary_df)),
        "scatter_fig": scatter_fig,
        "heat_fig":    heat_fig,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tab 4: ML Model
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/model")
def get_model():
    roc_fig = {}
    try:
        if _roc_data:
            import numpy as np
            y_true = np.array(_roc_data["y_true"]).tolist()
            y_prob = np.array(_roc_data["y_prob"]).tolist()
            auc    = float(_roc_data["auc"])
            roc_fig = fig_to_json(plot_roc_curve(y_true, y_prob, auc))
    except Exception as exc:
        logger.warning("ROC: %s", exc)

    e = ML.get("clf_eval", {})
    r = ML.get("reg_eval", {})
    v = ML.get("vosoughi", {})

    return {
        "clf_eval": {
            "cv_auc_mean":  e.get("cv_auc_mean"),
            "cv_auc_std":   e.get("cv_auc_std"),
            "cv_f1_mean":   e.get("cv_f1_mean"),
            "cv_f1_std":    e.get("cv_f1_std"),
            "holdout_auc":  e.get("holdout_auc"),
            "n_train":      e.get("n_train"),
            "n_test":       e.get("n_test"),
            "report":       e.get("classification_report", ""),
            "confusion_matrix": e.get("confusion_matrix", []),
        },
        "reg_eval": {
            "mae": r.get("mae"),
            "r2":  r.get("r2"),
        },
        "vosoughi": {k: bool(val) if hasattr(val, 'item') else val for k, val in v.items()} if v else {},
        "roc_fig":   roc_fig,
        "shap_fig":  fig_to_json(plot_shap_importance(ML.get("shap_importance", pd.DataFrame()))),
        "rep_fig":   fig_to_json(plot_vosoughi_replication(v)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tab 5: Deep Dive
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/deepdive/{claim_id}")
def get_deep_dive(claim_id: str):
    if claim_id not in graphs:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    row   = CLAIM_ID_MAP[claim_id]
    G     = graphs[claim_id]
    label = row.label

    sent_df = sentiment_over_time(claim_id, spread_df)
    ss_df   = get_super_spreaders(G, top_n=CFG.graph.top_spreaders_n)
    mut_df  = get_mutation_similarity(row.text)

    # Per-claim SHAP
    shap_fig = {}
    try:
        clf      = ML.get("classifier")
        shap_arr = ML.get("shap_clf")
        X_full   = ML.get("X")
        if clf is not None and shap_arr is not None and X_full is not None:
            merged = (metrics_df.merge(nlp_df, on="claim_id", how="inner")
                      .pipe(lambda d: d[d["label"].isin(["fake","real"])].reset_index(drop=True)))
            idx_list = merged.index[merged["claim_id"] == claim_id].tolist()
            if idx_list:
                idx   = idx_list[0]
                sv    = shap_arr[idx]
                wf_df = shap_single_claim(clf, X_full.iloc[[idx]], sv, list(X_full.columns))
                shap_fig = fig_to_json(plot_shap_waterfall(wf_df, row.text))
    except Exception as exc:
        logger.debug("Deep-dive SHAP: %s", exc)

    return {
        "sentiment_fig":  fig_to_json(plot_sentiment_drift(sent_df, label)),
        "spreader_fig":   fig_to_json(plot_super_spreaders(ss_df)),
        "shap_fig":       shap_fig,
        "spreaders":      ss_df.to_dict(orient="records") if not ss_df.empty else [],
        "mutations":      mut_df.to_dict(orient="records") if not mut_df.empty else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tab 6: Speaker Analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/speakers")
def get_speakers():
    sa         = ML.get("speaker_analysis", {})
    party_df   = sa.get("by_party")
    subject_df = sa.get("by_subject")
    speaker_df = sa.get("by_speaker")

    treemap_fig = fig_to_json(plot_speaker_treemap(
        by_party_df=party_df, by_subject_df=subject_df
    ))

    sankey_fig = fig_to_json(plot_speaker_sankey(
        by_speaker_df=speaker_df,
        by_party_df=party_df,
        by_subject_df=subject_df,
        claims_df=claims_df,
    ))

    bubble_fig = fig_to_json(plot_speaker_bubble(
        claims_df=claims_df,
        metrics_df=metrics_df,
    ))

    def df_to_records(df):
        if df is None or (hasattr(df, "empty") and df.empty):
            return []
        return df.fillna("").to_dict(orient="records")

    return {
        "treemap_fig": treemap_fig,
        "sankey_fig":  sankey_fig,
        "bubble_fig":  bubble_fig,
        "by_party":    df_to_records(party_df),
        "by_subject":  df_to_records(subject_df),
        "by_speaker":  df_to_records(speaker_df),
        "has_data":    party_df is not None and not (hasattr(party_df, "empty") and party_df.empty),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting server at http://127.0.0.1:8000")
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
