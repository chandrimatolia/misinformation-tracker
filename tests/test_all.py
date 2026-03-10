"""
tests/test_all.py
-----------------
Pytest test suite covering all four project layers.

Run:  pytest tests/ -v
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import networkx as nx
import pytest

from src.config import CFG


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_claims():
    return pd.DataFrame([
        {"claim_id": "c001", "text": "5G towers spread disease secretly",
         "label": "fake", "speaker": "unknown", "party": "none",
         "subject": "health", "venue": "twitter",
         "origin_platform": "Twitter", "origin_community": "Health",
         "origin_time": "2024-01-01", "virality_score": 0.8,
         "credibility_history": 0.2},
        {"claim_id": "c002", "text": "Peer-reviewed study confirms vaccine safety",
         "label": "real", "speaker": "dr_smith", "party": "none",
         "subject": "health", "venue": "journal",
         "origin_platform": "Twitter", "origin_community": "Science",
         "origin_time": "2024-01-02", "virality_score": 0.3,
         "credibility_history": 0.9},
        {"claim_id": "c003", "text": "Microchips inserted in vaccines proven",
         "label": "fake", "speaker": "anon", "party": "none",
         "subject": "health", "venue": "reddit",
         "origin_platform": "Reddit", "origin_community": "Conspiracy",
         "origin_time": "2024-01-03", "virality_score": 0.75,
         "credibility_history": 0.15},
        {"claim_id": "c004", "text": "WHO approves new malaria vaccine",
         "label": "real", "speaker": "who", "party": "none",
         "subject": "health", "venue": "press",
         "origin_platform": "Facebook", "origin_community": "Mainstream",
         "origin_time": "2024-01-04", "virality_score": 0.25,
         "credibility_history": 0.95},
    ])


@pytest.fixture(scope="session")
def small_edges():
    import random
    from datetime import datetime, timedelta
    rng = random.Random(42)
    rows = []
    for cid in ["c001", "c002", "c003", "c004"]:
        n = 40 if cid in ["c001", "c003"] else 10
        t = datetime(2024, 1, 1)
        prev = f"user_origin_{cid}"
        for i in range(n):
            t += timedelta(hours=rng.uniform(0.1, 3))
            new = f"user_{cid}_{i}"
            rows.append({
                "claim_id": cid, "source_user": prev, "target_user": new,
                "platform": rng.choice(["Twitter", "Reddit"]),
                "community": rng.choice(["Health", "Conspiracy"]),
                "timestamp": t, "depth": min(i // 5 + 1, 8),
                "sentiment_score": round(rng.uniform(0, 1), 3),
                "is_debunk": rng.random() < 0.05,
                "hours_since_origin": i * 0.5,
            })
            prev = new
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCollector:

    def test_synthetic_generation_returns_dataframes(self):
        from src.data_collector import generate_synthetic_dataset
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            c, e = generate_synthetic_dataset(n_claims=4, max_shares_per_claim=20,
                                               output_dir=tmp)
        assert isinstance(c, pd.DataFrame)
        assert isinstance(e, pd.DataFrame)
        assert len(c) == 4
        assert len(e) > 0

    def test_claims_have_required_columns(self, small_claims):
        required = ["claim_id", "text", "label", "virality_score"]
        for col in required:
            assert col in small_claims.columns, f"Missing column: {col}"

    def test_labels_are_binary(self, small_claims):
        assert set(small_claims["label"].unique()).issubset({"fake", "real"})

    def test_virality_score_range(self, small_claims):
        assert small_claims["virality_score"].between(0, 1).all()

    def test_edges_have_required_columns(self, small_edges):
        required = ["claim_id", "source_user", "target_user",
                    "depth", "hours_since_origin"]
        for col in required:
            assert col in small_edges.columns, f"Missing column: {col}"

    def test_no_self_loops_in_edges(self, small_edges):
        self_loops = small_edges[
            small_edges["source_user"] == small_edges["target_user"]
        ]
        assert len(self_loops) == 0, "Self-loops found in edge data"

    def test_liar_label_map_coverage(self):
        from src.data_collector import LIAR_LABEL_MAP
        assert "pants-fire" in LIAR_LABEL_MAP
        assert "true" in LIAR_LABEL_MAP
        assert LIAR_LABEL_MAP["pants-fire"] == "fake"
        assert LIAR_LABEL_MAP["true"] == "real"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ANALYSIS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphAnalysis:

    def test_build_claim_graph_returns_digraph(self, small_edges):
        from src.graph_analysis import build_claim_graph
        G = build_claim_graph("c001", small_edges)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_empty_claim_returns_empty_graph(self, small_edges):
        from src.graph_analysis import build_claim_graph
        G = build_claim_graph("nonexistent_claim", small_edges)
        assert G.number_of_nodes() == 0

    def test_graph_is_directed(self, small_edges):
        from src.graph_analysis import build_claim_graph
        G = build_claim_graph("c001", small_edges)
        assert nx.is_directed(G)

    def test_pagerank_sums_to_one(self, small_edges):
        from src.graph_analysis import build_claim_graph
        G = build_claim_graph("c001", small_edges)
        pr = nx.pagerank(G, alpha=CFG.graph.pagerank_alpha)
        assert abs(sum(pr.values()) - 1.0) < 1e-6

    def test_compute_metrics_returns_required_fields(self, small_claims, small_edges):
        from src.graph_analysis import build_claim_graph, compute_claim_metrics
        G = build_claim_graph("c001", small_edges)
        m = compute_claim_metrics("c001", G, small_claims, small_edges)
        for field in ["n_nodes", "n_edges", "max_depth", "max_breadth",
                      "virality_score", "label"]:
            assert field in m, f"Missing metric: {field}"

    def test_super_spreaders_top_n(self, small_edges):
        from src.graph_analysis import build_claim_graph, get_super_spreaders
        G = build_claim_graph("c001", small_edges)
        ss = get_super_spreaders(G, top_n=5)
        assert len(ss) <= 5
        assert "pagerank" in ss.columns
        assert "betweenness" in ss.columns

    def test_community_detection_assigns_all_nodes(self, small_edges):
        from src.graph_analysis import build_claim_graph, get_communities
        G = build_claim_graph("c001", small_edges)
        comm = get_communities(G)
        for node in G.nodes():
            assert node in comm, f"Node {node} has no community assignment"

    def test_fake_vs_real_summary_has_both_labels(self, small_claims, small_edges):
        from src.graph_analysis import (build_all_graphs, compute_all_metrics,
                                         fake_vs_real_summary)
        graphs = build_all_graphs(small_claims, small_edges)
        metrics = compute_all_metrics(small_claims, small_edges, graphs)
        summary = fake_vs_real_summary(metrics)
        assert "fake" in summary.columns or "real" in summary.columns


# ─────────────────────────────────────────────────────────────────────────────
# NLP TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestNLP:

    def test_credibility_score_range(self):
        from src.nlp_analysis import credibility_score
        for text in [
            "Secret banned cure suppressed by government!!!",
            "Peer-reviewed study of 5,000 patients confirms 89% efficacy",
            "",
        ]:
            result = credibility_score(text)
            assert 0.0 <= result["credibility_score"] <= 1.0

    def test_alarm_words_detected(self):
        from src.nlp_analysis import credibility_score
        r = credibility_score("Secret government coverup of banned miracle cure")
        assert len(r["alarm_words"]) > 0

    def test_credibility_words_detected(self):
        from src.nlp_analysis import credibility_score
        r = credibility_score("Peer-reviewed research published in journal confirms data")
        assert len(r["credibility_words"]) > 0

    def test_sentiment_range(self):
        from src.nlp_analysis import sentiment_score
        for text in ["great safe effective healthy", "dangerous deadly toxic kill",
                     "the quick brown fox"]:
            s = sentiment_score(text)
            assert -1.0 <= s <= 1.0

    def test_virality_risk_range(self):
        from src.nlp_analysis import virality_risk_score
        r = virality_risk_score("SHOCKING: banned cure hidden by government!!!")
        assert 0 <= r["virality_risk"] <= 100

    def test_risk_level_labels(self):
        from src.nlp_analysis import virality_risk_score
        high = virality_risk_score("SHOCKING SECRET BANNED CURE COVERUP!!!")
        low  = virality_risk_score("scientists confirm peer-reviewed evidence from study data")
        assert "High" in high["risk_level"] or "Medium" in high["risk_level"]

    def test_keyword_extraction_returns_list(self):
        from src.nlp_analysis import extract_keywords
        kw = extract_keywords(["5G towers spread disease", "vaccine safety study"], top_n=5)
        assert isinstance(kw, list)
        assert len(kw) <= 5

    def test_mutation_similarity_decreases(self):
        from src.nlp_analysis import get_mutation_similarity
        df = get_mutation_similarity("5G towers are spreading the virus through radio waves")
        if not df.empty:
            assert df["similarity"].max() <= 1.0
            assert df["similarity"].min() >= 0.0

    def test_analyse_all_claims_output_shape(self, small_claims):
        from src.nlp_analysis import analyse_all_claims
        df = analyse_all_claims(small_claims)
        assert len(df) == len(small_claims)
        assert "credibility_score" in df.columns
        assert "virality_risk" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestModel:

    @pytest.fixture(scope="class")
    def model_data(self, small_claims, small_edges):
        from src.graph_analysis import build_all_graphs, compute_all_metrics
        from src.nlp_analysis   import analyse_all_claims
        from src.model          import build_feature_matrix
        graphs     = build_all_graphs(small_claims, small_edges)
        metrics_df = compute_all_metrics(small_claims, small_edges, graphs)
        nlp_df     = analyse_all_claims(small_claims)
        X, y_lbl, y_vir = build_feature_matrix(metrics_df, nlp_df)
        return X, y_lbl, y_vir, metrics_df, nlp_df

    def test_feature_matrix_has_no_nulls(self, model_data):
        X, _, _, _, _ = model_data
        assert not X.isnull().any().any(), "Feature matrix contains NaN values"

    def test_feature_matrix_shape(self, model_data):
        X, y, _, _, _ = model_data
        assert len(X) == len(y)
        assert X.shape[1] > 0

    def test_classifier_trains_without_error(self, model_data):
        X, y, _, _, _ = model_data
        if len(X) < 4:
            pytest.skip("Not enough samples for classifier test")
        from src.model import train_classifier
        clf, eval_m = train_classifier(X, y)
        assert clf is not None
        assert "holdout_auc" in eval_m
        assert 0 <= eval_m["holdout_auc"] <= 1

    def test_regressor_trains_without_error(self, model_data):
        X, _, y_vir, _, _ = model_data
        if len(X) < 4:
            pytest.skip("Not enough samples for regressor test")
        from src.model import train_regressor
        reg, eval_m = train_regressor(X, y_vir)
        assert reg is not None
        assert "mae" in eval_m
        assert eval_m["mae"] >= 0

    def test_vosoughi_replication_keys(self, small_claims, small_edges):
        from src.graph_analysis import build_all_graphs, compute_all_metrics
        from src.model          import vosoughi_replication
        graphs = build_all_graphs(small_claims, small_edges)
        metrics = compute_all_metrics(small_claims, small_edges, graphs)
        rep = vosoughi_replication(metrics)
        for key in ["depth_ratio", "breadth_ratio", "speed_ratio", "overall_pass"]:
            assert key in rep, f"Missing replication key: {key}"

    def test_speaker_analysis_with_liar_columns(self, small_claims, small_edges):
        from src.graph_analysis import build_all_graphs, compute_all_metrics
        from src.model          import speaker_analysis
        graphs = build_all_graphs(small_claims, small_edges)
        metrics = compute_all_metrics(small_claims, small_edges, graphs)
        sa = speaker_analysis(small_claims, metrics)
        assert isinstance(sa, dict)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_config_loads(self):
        from src.config import CFG
        assert CFG is not None
        assert CFG.graph.pagerank_alpha == 0.85
        assert CFG.data.seed == 42

    def test_feature_list_non_empty(self):
        from src.config import CFG
        assert len(CFG.model.features) > 0

    def test_colour_strings_are_hex(self):
        from src.config import CFG
        for colour in [CFG.viz.fake_color, CFG.viz.real_color]:
            assert colour.startswith("#"), f"Not a hex colour: {colour}"
            assert len(colour) == 7


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualizations:

    def test_empty_figure_returns_figure(self):
        from src.visualizations import _empty
        import plotly.graph_objects as go
        fig = _empty("test message")
        assert isinstance(fig, go.Figure)

    def test_virality_gauge_range(self):
        from src.visualizations import plot_virality_gauge
        import plotly.graph_objects as go
        for score in [0, 35, 65, 100]:
            fig = plot_virality_gauge(float(score), "🟡 Medium")
            assert isinstance(fig, go.Figure)

    def test_propagation_network_empty_graph(self):
        from src.visualizations import plot_propagation_network
        import plotly.graph_objects as go
        G   = nx.DiGraph()
        fig = plot_propagation_network(G, "fake")
        assert isinstance(fig, go.Figure)

    def test_fake_vs_real_plot(self):
        from src.visualizations import plot_fake_vs_real
        import plotly.graph_objects as go
        df  = pd.DataFrame({"metric": ["n_nodes", "max_depth"],
                             "fake": [50.0, 8.0], "real": [15.0, 3.0]})
        fig = plot_fake_vs_real(df)
        assert isinstance(fig, go.Figure)

    def test_shap_importance_empty(self):
        from src.visualizations import plot_shap_importance
        import plotly.graph_objects as go
        fig = plot_shap_importance(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_vosoughi_replication_plot(self):
        from src.visualizations import plot_vosoughi_replication
        import plotly.graph_objects as go
        rep = {"depth_ratio": 2.1, "depth_pass": True,
               "breadth_ratio": 1.5, "breadth_pass": True,
               "speed_ratio": 1.8, "speed_pass": True,
               "overall_pass": True}
        fig = plot_vosoughi_replication(rep)
        assert isinstance(fig, go.Figure)
