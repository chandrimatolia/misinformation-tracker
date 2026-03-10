from __future__ import annotations
"""
graph_analysis.py
-----------------
Builds propagation graphs from edge data and computes network metrics:
  - Cascade depth / breadth
  - Super-spreader detection (PageRank, betweenness centrality)
  - Community detection (Louvain)
  - Fake vs. real spread comparison
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain          # python-louvain


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILD GRAPH FOR A SINGLE CLAIM
# ─────────────────────────────────────────────────────────────────────────────

def build_claim_graph(
    claim_id: str,
    spread_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Returns a directed graph where nodes = users and edges = shares.
    Edge attributes: platform, community, timestamp, depth, sentiment_score, is_debunk.
    """
    edges = spread_df[spread_df["claim_id"] == claim_id].copy()
    if edges.empty:
        return nx.DiGraph()

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(
            row["source_user"],
            row["target_user"],
            platform=row.get("platform", "Unknown"),
            community=row.get("community", "Unknown"),
            timestamp=str(row.get("timestamp", "")),
            depth=int(row.get("depth", 0)),
            sentiment=float(row.get("sentiment_score", 0.5)),
            is_debunk=bool(row.get("is_debunk", False)),
            hours_since_origin=float(row.get("hours_since_origin", 0)),
        )

    return G


def build_all_graphs(
    claims_df: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> dict[str, nx.DiGraph]:
    """Returns {claim_id: DiGraph} for all claims."""
    graphs = {}
    for cid in claims_df["claim_id"]:
        graphs[cid] = build_claim_graph(cid, spread_df)
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-CLAIM METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_claim_metrics(
    claim_id: str,
    G: nx.DiGraph,
    claims_df: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> dict:
    """
    Returns a flat dict of network metrics for one claim.
    """
    if G.number_of_nodes() == 0:
        return {"claim_id": claim_id, "error": "empty graph"}

    claim_row = claims_df[claims_df["claim_id"] == claim_id].iloc[0]
    edges = spread_df[spread_df["claim_id"] == claim_id]

    # ── cascade shape ─────────────────────────────────────────────────────────
    max_depth   = int(edges["depth"].max()) if not edges.empty else 0
    max_breadth = int(
        edges.groupby("depth")["target_user"].nunique().max()
    ) if not edges.empty else 0

    # ── centrality (PageRank on undirected copy for speed) ───────────────────
    UG = G.to_undirected()
    try:
        pr    = nx.pagerank(G, alpha=0.85, max_iter=200)
        top_pr_node = max(pr, key=pr.get)
        top_pr_score = round(pr[top_pr_node], 5)
    except Exception:
        top_pr_node, top_pr_score = "N/A", 0.0

    # ── community detection ───────────────────────────────────────────────────
    try:
        partition = community_louvain.best_partition(UG)
        n_communities = len(set(partition.values()))
        modularity = round(
            community_louvain.modularity(partition, UG), 4
        )
    except Exception:
        n_communities, modularity = 1, 0.0

    # ── debunking ─────────────────────────────────────────────────────────────
    n_debunk   = int(edges["is_debunk"].sum()) if not edges.empty else 0
    debunk_pct = round(n_debunk / max(len(edges), 1) * 100, 2)

    # ── speed (median hours between first 10 shares) ──────────────────────────
    early = edges.nsmallest(10, "hours_since_origin")["hours_since_origin"]
    median_speed = round(float(early.median()) if len(early) > 1 else 0.0, 2)

    # ── sentiment drift ───────────────────────────────────────────────────────
    if not edges.empty and "sentiment_score" in edges.columns:
        sent = edges.sort_values("hours_since_origin")["sentiment_score"]
        sentiment_drift = round(float(sent.iloc[-1] - sent.iloc[0]), 4)
    else:
        sentiment_drift = 0.0

    return {
        "claim_id":          claim_id,
        "label":             claim_row["label"],
        "text":              claim_row["text"][:80] + "…",
        "n_nodes":           G.number_of_nodes(),
        "n_edges":           G.number_of_edges(),
        "max_depth":         max_depth,
        "max_breadth":       max_breadth,
        "n_communities":     n_communities,
        "modularity":        modularity,
        "top_spreader":      top_pr_node,
        "top_pagerank":      top_pr_score,
        "debunk_count":      n_debunk,
        "debunk_pct":        debunk_pct,
        "median_speed_hrs":  median_speed,
        "sentiment_drift":   sentiment_drift,
        "virality_score":    float(claim_row.get("virality_score", 0)),
    }


def compute_all_metrics(
    claims_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    graphs: dict[str, nx.DiGraph],
) -> pd.DataFrame:
    rows = []
    for cid, G in graphs.items():
        rows.append(compute_claim_metrics(cid, G, claims_df, spread_df))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUPER-SPREADER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def get_super_spreaders(
    G: nx.DiGraph, top_n: int = 10
) -> pd.DataFrame:
    """Returns top-N nodes ranked by PageRank with extra centrality metrics."""
    if G.number_of_nodes() < 2:
        return pd.DataFrame()

    pr  = nx.pagerank(G, alpha=0.85, max_iter=200)
    deg = dict(G.out_degree())

    try:
        btw = nx.betweenness_centrality(G, normalized=True, k=min(100, G.number_of_nodes()))
    except Exception:
        btw = {n: 0.0 for n in G.nodes()}

    rows = [
        {
            "user": n,
            "pagerank":    round(pr.get(n, 0), 6),
            "out_degree":  deg.get(n, 0),
            "betweenness": round(btw.get(n, 0), 6),
        }
        for n in G.nodes()
    ]
    df = pd.DataFrame(rows).sort_values("pagerank", ascending=False).head(top_n)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMMUNITY DETECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_communities(G: nx.DiGraph) -> dict[str, int]:
    """Returns {node: community_id} partition dict."""
    UG = G.to_undirected()
    if UG.number_of_nodes() < 2:
        return {n: 0 for n in UG.nodes()}
    return community_louvain.best_partition(UG)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FAKE vs REAL COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def fake_vs_real_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates key metrics by label (fake / real)."""
    cols = [
        "n_nodes", "n_edges", "max_depth", "max_breadth",
        "n_communities", "median_speed_hrs", "debunk_pct",
        "virality_score", "sentiment_drift",
    ]
    available = [c for c in cols if c in metrics_df.columns]
    summary = (
        metrics_df.groupby("label")[available]
        .mean()
        .round(3)
        .T
        .rename_axis("metric")
        .reset_index()
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_collector import load_dataset

    claims_df, spread_df = load_dataset()
    graphs      = build_all_graphs(claims_df, spread_df)
    metrics_df  = compute_all_metrics(claims_df, spread_df, graphs)

    print("\n── Metrics sample ──")
    print(metrics_df[["claim_id", "label", "n_nodes", "max_depth", "virality_score"]].head())

    print("\n── Fake vs Real ──")
    print(fake_vs_real_summary(metrics_df))
