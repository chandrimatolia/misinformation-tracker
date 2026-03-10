from __future__ import annotations
"""
src/config.py
-------------
Loads config.yaml and exposes a typed CFG object.
All modules import constants from here instead of hard-coding them.

Usage
-----
    from src.config import CFG
    alpha = CFG.graph.pagerank_alpha
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)

# ── locate config.yaml relative to this file ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


@dataclass
class DataConfig:
    dir: str = "data"
    liar_subdir: str = "liar"
    max_claims: int = 200
    max_shares_per_claim: int = 200
    seed: int = 42


@dataclass
class GraphConfig:
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 200
    max_nodes_display: int = 200
    betweenness_k: int = 100
    top_spreaders_n: int = 10


@dataclass
class NLPConfig:
    tfidf_max_features: int = 300
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    top_keywords_n: int = 15
    credibility_threshold: float = 0.45


@dataclass
class ModelConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    seed: int = 42
    features: List[str] = field(default_factory=lambda: [
        "n_nodes", "n_edges", "max_depth", "max_breadth",
        "n_communities", "modularity", "median_speed_hrs",
        "debunk_pct", "sentiment_drift", "credibility_score",
        "virality_risk",
    ])


@dataclass
class VizConfig:
    fake_color: str = "#EF4444"
    real_color: str = "#22C55E"
    neutral_color: str = "#6366F1"
    background: str = "#0F172A"
    grid_color: str = "#1E293B"
    text_color: str = "#E2E8F0"
    animation_frame_ms: int = 100


@dataclass
class DashboardConfig:
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt: str = "%H:%M:%S"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _nested_update(dc: object, d: dict) -> None:
    """Recursively populate dataclass fields from dict."""
    for k, v in d.items():
        if hasattr(dc, k):
            attr = getattr(dc, k)
            if isinstance(v, dict) and hasattr(attr, "__dataclass_fields__"):
                _nested_update(attr, v)
            else:
                setattr(dc, k, v)


def load_config(path: Path = _CONFIG_PATH) -> Config:
    cfg = Config()
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _nested_update(cfg, raw)
        logger.debug("Config loaded from %s", path)
    else:
        logger.warning("config.yaml not found at %s — using defaults", path)
    return cfg


def setup_logging(cfg: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format=cfg.format,
        datefmt=cfg.datefmt,
    )


# ── module-level singleton ────────────────────────────────────────────────────
CFG = load_config()
setup_logging(CFG.logging)
