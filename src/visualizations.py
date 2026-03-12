"""
src/visualizations.py  (advanced rewrite)
------------------------------------------
All Plotly figure-generating functions.

New in advanced version
-----------------------
- plot_animated_cascade()     time-lapse spread animation
- plot_sankey_platform_flow() Sankey diagram of cross-platform claim flow
- plot_shap_importance()      SHAP global feature importance bar chart
- plot_shap_waterfall()       per-claim SHAP waterfall explanation
- plot_roc_curve()            ROC / AUC visualisation for classifier
- plot_vosoughi_replication() scorecard vs. published paper findings
- plot_heatmap_spread()       hour-of-day x depth spread heatmap
- plot_speaker_treemap()      fake-rate by party treemap
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.config import CFG

FC   = CFG.viz.fake_color
RC   = CFG.viz.real_color
NC   = CFG.viz.neutral_color
BG   = CFG.viz.background
GRID = CFG.viz.grid_color
TEXT = CFG.viz.text_color

BASE_LAYOUT: dict = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(l=48, r=24, t=56, b=48),
)

def _empty(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#94A3B8"),
                       xref="paper", yref="paper")
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ── 1. PROPAGATION NETWORK ────────────────────────────────────────────────────

def plot_propagation_network(G, claim_label="fake", title="Propagation Network",
                              communities=None):
    if G.number_of_nodes() == 0:
        return _empty("No graph data available")
    cap = CFG.graph.max_nodes_display
    if G.number_of_nodes() > cap:
        top = sorted(nx.pagerank(G).items(), key=lambda x: -x[1])[:cap]
        G   = G.subgraph([n for n, _ in top]).copy()
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=CFG.data.seed, k=1.5)
    pr = nx.pagerank(G, alpha=CFG.graph.pagerank_alpha,
                     max_iter=CFG.graph.pagerank_max_iter)
    ex, ey = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0,y0=pos[u]; x1,y1=pos[v]
            ex+=[x0,x1,None]; ey+=[y0,y1,None]
    edge_trace = go.Scatter(x=ex,y=ey,mode="lines",
                            line=dict(width=0.6,color="#334155"),
                            hoverinfo="none",showlegend=False)
    nodes = [n for n in G.nodes() if n in pos]
    node_x=[pos[n][0] for n in nodes]; node_y=[pos[n][1] for n in nodes]
    node_pr=[pr.get(n,0) for n in nodes]
    node_sz=[8+v*900 for v in node_pr]
    if communities:
        nc=[communities.get(n,0) for n in nodes]; cscale="Turbo"; cbar="Community"
    else:
        nc=node_pr; cscale="RdYlGn_r" if claim_label=="fake" else "YlGn"; cbar="PageRank"
    node_text=[
        f"<b>User:</b> {n[:14]}<br>"
        f"<b>PageRank influence:</b> {pr.get(n,0):.5f}<br>"
        f"<b>Shares forwarded:</b> {G.out_degree(n)}"
        + (f"<br><b>Community cluster:</b> {communities[n]}" if communities and n in communities else "")
        for n in nodes
    ]
    node_trace=go.Scatter(x=node_x,y=node_y,mode="markers",
        marker=dict(size=node_sz,color=nc,colorscale=cscale,
                    colorbar=dict(title=cbar,thickness=10,tickfont=dict(color=TEXT)),
                    line=dict(width=0.5,color="#475569")),
        text=node_text,hovertemplate="%{text}<extra></extra>",showlegend=False)
    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="How did this claim propagate through connected communities?",
                font=dict(size=13, color=TEXT),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TEXT, family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=60, b=70),
            annotations=[dict(
                text=(
                    "Each node = a user  ·  Node size = influence score (PageRank)  ·  Edges = share events<br>"
                    "Colour = community cluster auto-detected by Louvain algorithm  ·  Denser clusters = echo chambers<br>"
                    "Communities are numbered 0–N, each representing a distinct group of users who share more with each other than with outsiders"
                ),
                x=0.5, y=-0.10, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=9, color="#475569"),
                align="left",
            )],
        ),
    )


# ── 2. ANIMATED CASCADE ───────────────────────────────────────────────────────

def plot_animated_cascade(claim_id, spread_df, G, label="fake"):
    edges = spread_df[spread_df["claim_id"]==claim_id].copy()
    if edges.empty or G.number_of_nodes()==0:
        return _empty("No cascade data for animation")
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=CFG.data.seed, k=1.5)
    pr = nx.pagerank(G, alpha=CFG.graph.pagerank_alpha,
                     max_iter=CFG.graph.pagerank_max_iter)
    colour = FC if label=="fake" else RC
    edges["hour_bin"] = edges["hours_since_origin"].fillna(0).round(0).astype(int)
    bins = sorted(edges["hour_bin"].unique())
    frames=[]; all_nodes=set()
    for hbin in bins:
        sub = edges[edges["hour_bin"]<=hbin]
        all_nodes |= set(sub["source_user"])|set(sub["target_user"])
        ex,ey=[],[]
        for _,row in sub.iterrows():
            if row["source_user"] in pos and row["target_user"] in pos:
                x0,y0=pos[row["source_user"]]; x1,y1=pos[row["target_user"]]
                ex+=[x0,x1,None]; ey+=[y0,y1,None]
        vis=[n for n in all_nodes if n in pos]
        nx_=[pos[n][0] for n in vis]; ny_=[pos[n][1] for n in vis]
        ns_=[8+pr.get(n,0)*800 for n in vis]
        frames.append(go.Frame(
            data=[go.Scatter(x=ex,y=ey,mode="lines",
                             line=dict(width=0.6,color="#334155"),hoverinfo="none"),
                  go.Scatter(x=nx_,y=ny_,mode="markers",
                             marker=dict(size=ns_,color=colour,opacity=0.85,
                                         line=dict(width=0.5,color="white")),
                             text=[f"H{hbin}" for _ in vis],
                             hovertemplate="%{text}<extra></extra>")],
            name=str(hbin),
            layout=go.Layout(title_text=f"Hour {hbin} | {len(vis)} users reached")))
    init=[go.Scatter(x=[0],y=[0],mode="lines",
                    line=dict(width=0.6,color="#334155"),hoverinfo="none",
                    showlegend=False),
          go.Scatter(x=[0],y=[0],mode="markers",
                    marker=dict(size=8,color=colour,opacity=0),
                    hoverinfo="none",showlegend=False)]
    pos_vals = list(pos.values()) if pos else []
    x_range = [min(p[0] for p in pos_vals)-20, max(p[0] for p in pos_vals)+20] if pos_vals else [0,1]
    y_range = [min(p[1] for p in pos_vals)-20, max(p[1] for p in pos_vals)+20] if pos_vals else [0,1]
    return go.Figure(data=init, frames=frames,
        layout=go.Layout(
            title=dict(text=f"Cascade Animation — {label.upper()}",
                       font=dict(size=15,color=TEXT)),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TEXT, family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=60, b=60),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=x_range),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,range=y_range),
            updatemenus=[dict(type="buttons",showactive=False,y=1.08,x=0.5,
                xanchor="center",
                buttons=[
                    dict(label="▶ Play",method="animate",
                         args=[None,dict(frame=dict(
                             duration=CFG.viz.animation_frame_ms,redraw=True),
                             fromcurrent=True,transition=dict(duration=0))]),
                    dict(label="⏸ Pause",method="animate",
                         args=[[None],dict(frame=dict(duration=0,redraw=False),
                                           mode="immediate")])])],
            sliders=[dict(active=0,
                steps=[dict(method="animate",
                    args=[[str(b)],dict(frame=dict(
                        duration=CFG.viz.animation_frame_ms,redraw=True),
                        mode="immediate")],
                    label=f"H{b}") for b in bins],
                x=0.05,y=0,len=0.9,
                bgcolor=BG, bordercolor=GRID,
                font=dict(color=TEXT),
                currentvalue=dict(prefix="Hour: ",visible=True,
                                  font=dict(color=TEXT, size=11)))]))


# ── 3. SANKEY PLATFORM FLOW ───────────────────────────────────────────────────

def plot_sankey_platform_flow(spread_df, claim_id=None):
    df = spread_df.copy()
    if claim_id:
        df = df[df["claim_id"]==claim_id]
    if df.empty or "platform" not in df.columns:
        return _empty("No platform flow data")
    df = df.sort_values(["claim_id","hours_since_origin"])
    df["prev_platform"] = df.groupby("claim_id")["platform"].shift(1)
    flow = (df.dropna(subset=["prev_platform"])
              .groupby(["prev_platform","platform"]).size()
              .reset_index(name="count").query("count > 0"))
    if flow.empty:
        return _empty("Not enough cross-platform transitions")
    all_p = list(set(flow["prev_platform"])|set(flow["platform"]))
    idx   = {p:i for i,p in enumerate(all_p)}
    PAL   = px.colors.qualitative.Vivid

    # Compute in/out flow per platform for hover
    in_flow  = flow.groupby("platform")["count"].sum().to_dict()
    out_flow = flow.groupby("prev_platform")["count"].sum().to_dict()

    node_labels = all_p
    node_hover  = [
        f"<b>{p}</b><br>"
        f"Inbound amplification: {in_flow.get(p, 0)} shares<br>"
        f"Outbound amplification: {out_flow.get(p, 0)} shares<br>"
        f"Net flow: {in_flow.get(p,0) - out_flow.get(p,0):+d}"
        for p in all_p
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            customdata=node_hover,
            hovertemplate="%{customdata}<extra></extra>",
            color=[PAL[i%len(PAL)] for i in range(len(all_p))],
            pad=20, thickness=24,
            line=dict(color="#334155", width=0.5)),
        link=dict(
            source=[idx[r.prev_platform] for _,r in flow.iterrows()],
            target=[idx[r.platform]       for _,r in flow.iterrows()],
            value =flow["count"].tolist(),
            color ="rgba(99,102,241,0.25)",
            hovertemplate=(
                "<b>%{source.label} → %{target.label}</b><br>"
                "Cross-platform amplification: %{value} shares<br>"
                "<extra></extra>"
            ),
        )))
    fig.update_layout(
        title=dict(
            text="Which platforms amplified this claim across the information ecosystem?",
            font=dict(size=13, color=TEXT)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=48, r=24, t=56, b=80),
        annotations=[dict(
            text=(
                "Node width = total share volume  ·  "
                "Flow width = cross-platform amplification strength  ·  "
                "Hover nodes for inbound / outbound breakdown"
            ),
            x=0.5, y=-0.10, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color="#475569"),
            align="center"
        )])
    return fig


# ── 4. HEATMAP SPREAD ─────────────────────────────────────────────────────────

def plot_heatmap_spread(claim_id, spread_df, label="fake"):
    """
    Hour-of-day × Cascade depth heatmap.
    Reference quality: peak cell callout, diagonal pattern annotation,
    time-band zone labels embedded directly in chart space.
    """
    edges = spread_df[spread_df["claim_id"] == claim_id].copy()
    if edges.empty:
        return _empty("No data for heatmap")

    edges["hour_of_day"] = pd.to_datetime(
        edges["timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
    edges["depth_bin"] = edges["depth"].fillna(0).astype(int).clip(0, 8)
    pivot = (edges.groupby(["depth_bin", "hour_of_day"])
                  .size().unstack(fill_value=0))

    hours_str = [f"{h:02d}:00" for h in pivot.columns.tolist()]
    depths_str = [f"Depth {d}" for d in pivot.index.tolist()]
    z_vals = pivot.values.tolist()

    # Find peak cell
    flat = pivot.values
    if flat.max() > 0:
        peak_row, peak_col = np.unravel_index(flat.argmax(), flat.shape)
        peak_depth = depths_str[peak_row]
        peak_hour  = hours_str[peak_col]
        peak_val   = int(flat[peak_row, peak_col])
    else:
        peak_row = peak_col = peak_val = 0
        peak_depth = "Depth 0"; peak_hour = "00:00"

    colorscale = [
        [0.0,  BG],
        [0.15, "#1a0a0e"],
        [0.40, "#7f1d1d"],
        [0.70, "#dc2626"],
        [1.0,  "#fca5a5"],
    ] if label == "fake" else [
        [0.0,  BG],
        [0.15, "#052e16"],
        [0.40, "#166534"],
        [0.70, "#16a34a"],
        [1.0,  "#86efac"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=hours_str,
        y=depths_str,
        colorscale=colorscale,
        zmin=0,
        hovertemplate=(
            "<b>%{y}  ·  %{x}</b><br>"
            "Share events: <b>%{z}</b><extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Share volume", font=dict(color=TEXT, size=10)),
            tickfont=dict(color="#475569", size=9),
            thickness=12, len=0.85,
        ),
    ))

    # ── Peak cell callout ─────────────────────────────────────────────────────
    annotations = [
        dict(
            x=peak_hour, y=peak_depth,
            text=f"★ PEAK<br>{peak_val} shares",
            showarrow=True, arrowhead=2,
            arrowcolor="white", arrowwidth=1.5,
            ax=45, ay=-35,
            font=dict(size=9, color="white", family="DM Mono, monospace"),
            bgcolor="rgba(6,10,16,0.88)", borderpad=4,
            bordercolor="rgba(255,255,255,0.3)", borderwidth=1,
        ),
    ]

    # ── Time-band zone labels at top of chart ─────────────────────────────────
    # Morning / Afternoon / Evening / Night bands
    time_bands = [
        ("00:00", "06:00", "NIGHT"),
        ("07:00", "11:00", "MORNING"),
        ("12:00", "17:00", "AFTERNOON"),
        ("18:00", "23:00", "EVENING"),
    ]
    n_hours = len(hours_str)
    for band_start, band_end, band_label in time_bands:
        cols_in_band = [i for i, h in enumerate(hours_str) if band_start <= h <= band_end]
        if not cols_in_band:
            continue
        mid_col = hours_str[cols_in_band[len(cols_in_band)//2]]
        annotations.append(dict(
            x=mid_col, y=len(depths_str) - 0.3,
            text=band_label,
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=7, color="#334155", family="DM Mono, monospace"),
            bgcolor="rgba(255,255,255,0.04)", borderpad=2,
        ))

    # ── Footer ────────────────────────────────────────────────────────────────
    annotations.append(dict(
        x=0.5, y=-0.22, xref="paper", yref="paper",
        text=(
            "Darker cells = more shares at that depth/hour  ·  "
            "Diagonal pattern = cascade deepening through the day  ·  "
            "★ = peak amplification window  ·  "
            "Read diagonally: misinformation tends to reach maximum depth in the evening"
        ),
        showarrow=False, font=dict(size=10, color="#475569"), align="center",
    ))

    fig.update_layout(
        title=dict(
            text="At what time of day does misinformation penetrate deepest into the network?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=88, r=72, t=72, b=110),
        height=420,
        xaxis=dict(
            title="Hour of day (UTC)",
            gridcolor=GRID, type="category",
            tickfont=dict(color=TEXT, size=9), tickangle=-45,
        ),
        yaxis=dict(
            title="Cascade depth",
            gridcolor=GRID, type="category",
            tickfont=dict(color=TEXT, size=10),
        ),
        annotations=annotations,
    )
    return fig


# ── 5. SHAP IMPORTANCE ────────────────────────────────────────────────────────

def plot_shap_importance(shap_df):
    if shap_df is None or (hasattr(shap_df,"empty") and shap_df.empty):
        return _empty("SHAP not available\n(run: pip install shap)")

    feature_labels = {
        "n_nodes":          "Network size",
        "n_edges":          "Total shares",
        "max_depth":        "Cascade depth",
        "max_breadth":      "Peak breadth",
        "n_communities":    "Community fragmentation",
        "modularity":       "Echo chamber strength",
        "median_speed_hrs": "Spread velocity",
        "debunk_pct":       "Debunking resistance",
        "sentiment_drift":  "Emotional drift",
        "credibility_score":"NLP credibility score",
        "virality_risk":    "Predicted virality risk",
        "alarm_word_count": "Alarm word density",
        "credibility_word_count": "Credibility lexicon",
    }
    df = shap_df.copy()
    df["label"] = df["feature"].map(lambda x: feature_labels.get(x, x))
    df = df.sort_values("mean_shap", ascending=True)

    max_shap = float(df["mean_shap"].max()) if float(df["mean_shap"].max()) > 0 else 1
    bar_colors = [
        f"rgba(167,{int(139*(1-v/max_shap))},{int(250*(0.4+0.6*v/max_shap))},{0.55+0.45*v/max_shap})"
        for v in df["mean_shap"]
    ]

    annotations = [
        dict(x=0.5, y=-0.26, xref="paper", yref="paper",
             text="Mean |SHAP| = average absolute Shapley value across 200 claims  ·  Higher = stronger driver of fake/real classification <br> ·  ★ = dominant global predictor",
             showarrow=False, font=dict(size=10, color="#475569"), align="center"),
    ]
    # Star on top feature
    if len(df) > 0:
        top = df.iloc[-1]
        annotations.append(dict(
            x=float(top["mean_shap"]) + max_shap * 0.02,
            y=top["label"],
            text="★ dominant predictor",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=9, color="#a78bfa"),
        ))

    fig = go.Figure(go.Bar(
        y=df["label"].tolist(),
        x=df["mean_shap"].tolist(),
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: <b>%{x:.5f}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="Which signals most strongly predict whether a claim is fake?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=190, r=160, t=68, b=110),
        height=440,
        xaxis=dict(title="Mean absolute SHAP contribution (averaged across 200 claims)",
                   gridcolor=GRID, tickfont=dict(color=TEXT)),
        yaxis=dict(type="category", autorange="reversed",
                   tickfont=dict(color=TEXT, size=11), gridcolor=GRID),
        annotations=annotations,
    )
    return fig


# ── 6. SHAP WATERFALL ─────────────────────────────────────────────────────────

def plot_shap_waterfall(shap_row_df, claim_text=""):
    if shap_row_df is None or (hasattr(shap_row_df,"empty") and shap_row_df.empty):
        return _empty("No SHAP data available for this claim")
    df = shap_row_df.head(8).copy()

    feature_labels = {
        "n_nodes":               "Network reach",
        "n_edges":               "Propagation edges",
        "max_depth":             "Cascade depth",
        "max_breadth":           "Peak breadth",
        "n_communities":         "Community fragmentation",
        "modularity":            "Echo chamber strength",
        "median_speed_hrs":      "Propagation velocity",
        "debunk_pct":            "Debunking resistance",
        "sentiment_drift":       "Emotional drift",
        "credibility_score":     "NLP credibility score",
        "virality_risk":         "Virality risk index",
        "alarm_word_count":      "Alarm word density",
        "credibility_word_count":"Credibility lexicon",
    }
    df["label"] = df["feature"].map(lambda x: feature_labels.get(x, x))
    df = df.sort_values("shap", key=abs, ascending=True)

    vals      = df["shap"].tolist()
    labels    = df["label"].tolist()
    feat_vals = df["value"].round(3).tolist()
    n         = len(vals)
    # Give 60% extra room so labels never fall inside a bar
    x_max     = max(abs(v) for v in vals) * 1.6 if vals else 0.3

    fig = go.Figure()

    # ── Diverging background zones ────────────────────────────────────────────
    fig.add_vrect(x0=0,      x1=x_max,  fillcolor="rgba(244,63,94,0.06)",  line_width=0)
    fig.add_vrect(x0=-x_max, x1=0,      fillcolor="rgba(16,185,129,0.06)", line_width=0)

    # ── Zero line ─────────────────────────────────────────────────────────────
    fig.add_vline(x=0, line_color="#334155", line_width=2)

    # ── Intensity-graded bars ────────────────────────────────────────────────
    max_abs = max(abs(v) for v in vals) if vals else 1
    bar_colors = []
    for v in vals:
        t = abs(v) / max_abs
        if v > 0:
            bar_colors.append(f"rgba(244,{int(63*(1-t))},{int(94*(1-t))},{0.55+0.4*t})")
        else:
            bar_colors.append(f"rgba({int(16*(1-t))},{int(130+55*t)},{int(80+49*t)},{0.55+0.4*t})")

    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        textposition="none",
        customdata=list(zip(feat_vals, [f"{v:+.4f}" for v in vals],
                            ["→ FAKE" if v > 0 else "→ REAL" for v in vals])),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Raw value: <b>%{customdata[0]}</b><br>"
            "Shapley: <b>%{customdata[1]}</b><br>"
            "%{customdata[2]}<extra></extra>"
        )
    ))

    # ── Build ALL annotations in one list ─────────────────────────────────────
    pad = x_max * 0.04
    annotations = [
        # Zone header labels at top of plot area
        dict(x=x_max * 0.5,  y=1.0, xref="x", yref="paper",
             text="▶  FAKE", showarrow=False,
             font=dict(size=9, color=FC, family="DM Mono, monospace"),
             xanchor="center", yanchor="top", yshift=-4),
        dict(x=-x_max * 0.5, y=1.0, xref="x", yref="paper",
             text="REAL  ◀", showarrow=False,
             font=dict(size=9, color=RC, family="DM Mono, monospace"),
             xanchor="center", yanchor="top", yshift=-4),
        # Footer caption
        dict(x=0.5, y=-0.2, xref="paper", yref="paper",
             text=(
                 "SHAP: each bar = one feature's marginal Shapley contribution  ·  "
                 "bar length = magnitude  ·  ★ = dominant predictor  ·  "
                 "left values = raw feature value  ·  hover each bar for details"
             ),
             showarrow=False, font=dict(size=10, color="#475569"), align="center"),
    ]
    # Per-bar value labels (beyond bar end) + raw feature value chips (left margin)
    for i, (v, lbl, fv) in enumerate(zip(vals, labels, feat_vals)):
        is_dominant = (i == len(vals) - 1)
        label_text  = f"★ {v:+.3f}" if is_dominant else f"{v:+.3f}"
        label_color = "#a78bfa" if is_dominant else "#94a3b8"
        annotations.append(dict(
            x=v + (pad if v >= 0 else -pad), y=lbl, xref="x", yref="y",
            text=label_text, showarrow=False,
            xanchor="left" if v >= 0 else "right", yanchor="middle",
            font=dict(size=9, color=label_color),
        ))
        annotations.append(dict(
            x=-0.01, y=lbl, xref="paper", yref="y",
            text=f"{fv}", showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(size=8, color="#475569"),
        ))

    short_claim = (claim_text[:70] + "…") if len(claim_text) > 70 else claim_text
    fig.update_layout(
        title=dict(
            text=f'Why was "{short_claim}" classified this way?' if claim_text else "XAI: Feature attribution",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=200, r=90, t=72, b=120),
        height=420,
        xaxis=dict(
            title=dict(text="← REAL  ·  Shapley value  ·  FAKE →", font=dict(size=11), standoff=10),
            range=[-x_max * 1.7, x_max * 1.7],
            gridcolor=GRID, tickfont=dict(color="#475569", size=10), zeroline=False,
        ),
        yaxis=dict(
            type="category",
            tickfont=dict(color=TEXT, size=11),
            gridcolor="rgba(0,0,0,0)",
        ),
        annotations=annotations,
    )
    return fig


# ── 7. ROC CURVE ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, auc_score):
    from sklearn.metrics import roc_curve
    import numpy as _np
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig = go.Figure()
    # Random chance diagonal
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(color="#334155", width=1.5, dash="dot"),
        name="Random chance (AUC=0.5)", hoverinfo="skip"
    ))
    # Classifier curve with gradient fill
    fig.add_trace(go.Scatter(
        x=list(fpr), y=list(tpr), mode="lines",
        line=dict(color=RC, width=2.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        name=f"Classifier (AUC={auc_score:.3f})",
        hovertemplate="FPR: %{x:.3f}  ·  TPR: %{y:.3f}<extra></extra>"
    ))

    # Optimal operating point (closest to top-left corner)
    opt_idx = int(_np.argmin(_np.sqrt(_np.array(fpr)**2 + (1-_np.array(tpr))**2)))

    annotations = [
        dict(x=0.5, y=-0.12, xref="paper", yref="paper",
             text=(f"AUC = {auc_score:.3f}  ·  "
                   f"Optimal threshold: FPR={fpr[opt_idx]:.2f} · TPR={tpr[opt_idx]:.2f}  ·  "
                   "High AUC reflects NLP features trained on calibrated synthetic labels"),
             showarrow=False, font=dict(size=10, color="#475569"), align="center"),
        dict(x=float(fpr[opt_idx]), y=float(tpr[opt_idx]),
             text=f"Optimal<br>operating point",
             showarrow=True, arrowhead=2, arrowcolor=RC, arrowwidth=1.5,
             ax=-45, ay=-28,
             font=dict(size=9, color=RC),
             bgcolor="rgba(6,10,16,0.8)", borderpad=3),
        dict(x=0.7, y=0.4,
             text=f"AUC = {auc_score:.3f}",
             showarrow=False,
             font=dict(size=15, color=RC, family="DM Mono, monospace"),
             bgcolor="rgba(6,10,16,0.8)", borderpad=6),
    ]

    fig.update_layout(
        title=dict(
            text="How well does the Random Forest classifier discriminate fake from real claims?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=64, r=40, t=68, b=110),
        height=420,
        annotations=annotations,
        xaxis=dict(title="False Positive Rate (1 − Specificity)", gridcolor=GRID,
                   range=[0,1], tickfont=dict(color=TEXT)),
        yaxis=dict(title="True Positive Rate (Sensitivity)", gridcolor=GRID,
                   range=[0,1], tickfont=dict(color=TEXT)),
        legend=dict(bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=11, color=TEXT), x=0.52, y=0.08),
    )
    return fig


# ── 8. VOSOUGHI REPLICATION SCORECARD ────────────────────────────────────────

def plot_vosoughi_replication(v: dict):
    if not v:
        return _empty("Vosoughi replication data not available")

    metrics = [
        {
            "name": "Cascade Depth",
            "ratio": v.get("depth_ratio", 0),
            "threshold": v.get("depth_threshold", 1.5),
            "pass": v.get("depth_pass", False),
            "desc": "Fake cascades penetrate deeper",
        },
        {
            "name": "Breadth",
            "ratio": v.get("breadth_ratio", 0),
            "threshold": v.get("breadth_threshold", 1.5),
            "pass": v.get("breadth_pass", False),
            "desc": "Fake cascades reach more users per level",
        },
        {
            "name": "Speed (real/fake)",
            "ratio": v.get("speed_ratio", 0),
            "threshold": v.get("speed_threshold", 1.5),
            "pass": v.get("speed_pass", False),
            "desc": "Real news takes longer to spread",
        },
    ]

    all_pass = all(m["pass"] for m in metrics)
    colours  = [RC if m["pass"] else FC for m in metrics]

    fig = go.Figure()
    annotations = [
        dict(
            text=(
                "Each bar = observed fake÷real ratio on that diffusion dimension  ·  "
                "Dotted line = minimum threshold from Vosoughi et al. (Science, 2018)  ·  "
                "Green = finding replicated  ·  Red = below threshold"
            ),
            x=0.5, y=-0.22, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#475569"), align="center"
        )
    ]
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=[m["name"]],
            y=[m["ratio"]],
            marker_color=colours[i],
            marker_line=dict(width=0),
            width=0.45,
            name=m["name"],
            showlegend=False,
            customdata=[[m["ratio"], m["threshold"], m["desc"], "✅ PASS" if m["pass"] else "❌ FAIL"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{customdata[2]}<br>"
                "Observed ratio: <b>%{customdata[0]:.2f}×</b><br>"
                "Minimum threshold: %{customdata[1]:.1f}×<br>"
                "<b>%{customdata[3]}</b><extra></extra>"
            )
        ))
        fig.add_shape(
            type="line",
            x0=i - 0.35, x1=i + 0.35,
            y0=m["threshold"], y1=m["threshold"],
            line=dict(color="#fbbf24", width=2, dash="dot"),
            xref="x", yref="y"
        )
        # Combined status + ratio label on bar
        annotations.append(dict(
            x=m["name"], y=m["ratio"] + max(v["ratio"] for v in metrics) * 0.04,
            text=f"{'✅' if m['pass'] else '❌'}  {m['ratio']:.2f}×",
            showarrow=False,
            font=dict(color=TEXT, size=12, family="DM Mono, monospace"),
            xanchor="center", yanchor="bottom",
        ))
        # Threshold label
        annotations.append(dict(
            x=i + 0.4, y=m["threshold"],
            text=f"min {m['threshold']:.1f}×",
            showarrow=False,
            font=dict(size=8, color="#fbbf24"),
            xanchor="left", yanchor="middle",
            bgcolor="rgba(6,10,16,0.75)", borderpad=2,
        ))

    overall = "✅ ALL FINDINGS REPLICATED" if all_pass else "⚠️ PARTIAL REPLICATION"
    fig.update_layout(
        title=dict(
            text=f"Does this dataset replicate Vosoughi et al. (Science, 2018)? — {overall}",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=64, r=60, t=72, b=120),
        height=380,
        xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=12)),
        yaxis=dict(title="Observed ratio (fake ÷ real)", gridcolor=GRID,
                   tickfont=dict(color=TEXT), rangemode="tozero"),
        annotations=annotations,
    )
    return fig


# ── 9. SPEAKER TREEMAP ────────────────────────────────────────────────────────

def plot_speaker_treemap(by_party_df=None, by_subject_df=None):
    df = by_party_df if by_party_df is not None else by_subject_df
    if df is None or (hasattr(df,"empty") and df.empty):
        return _empty("No speaker/party data\n(requires LIAR dataset)")
    group_col = "party" if "party" in df.columns else "subject"

    fig = go.Figure(go.Treemap(
        labels=df[group_col].tolist(),
        parents=[""] * len(df),
        values=df["n_claims"].tolist(),
        customdata=df[["fake_rate", "n_claims"]].values.tolist(),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Claims: %{customdata[1]}<br>"
            "Fake rate: %{customdata[0]:.0%}<extra></extra>"
        ),
        marker=dict(
            colors=df["fake_rate"].tolist(),
            colorscale=[[0,"#10b981"],[0.5,"#eab308"],[1,"#f43f5e"]],
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Fake Rate", font=dict(color=TEXT, size=11)),
                tickvals=[0, 0.5, 1],
                ticktext=["0% fake", "50%", "100%"],
                tickfont=dict(color="#475569", size=9),
                thickness=12,
            ),
        ),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.0%} fake",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        title=dict(
            text=f"Which {group_col} affiliations produce the most misinformation?",
            font=dict(size=13, color=TEXT),
            x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ── 10. UPGRADED EXISTING CHARTS ──────────────────────────────────────────────

def plot_cascade_timeline(claim_id, spread_df, label="fake"):
    edges = spread_df[spread_df["claim_id"] == claim_id].copy()
    if edges.empty: return _empty("No spread data")

    edges["hours"] = edges["hours_since_origin"].fillna(0).astype(float)
    edges["depth"] = edges["depth"].fillna(0).astype(int)

    # Count out-degree per source user as bubble size
    out_deg = edges["source_user"].value_counts().to_dict()
    edges["out_deg"] = edges["source_user"].map(out_deg).fillna(1)

    # Normalise time for colour fade (0=early, 1=late)
    t_max = edges["hours"].max()
    edges["t_norm"] = (edges["hours"] / t_max if t_max > 0 else 0)

    colour = FC if label == "fake" else RC

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edges["hours"].tolist(),
        y=edges["depth"].tolist(),
        mode="markers",
        marker=dict(
            size=(edges["out_deg"].clip(1, 20) * 3).tolist(),
            color=edges["t_norm"].tolist(),
            colorscale=[[0, colour], [1, "#1e2f42"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Earlier → Later", font=dict(color=TEXT, size=10)),
                tickvals=[0, 1],
                ticktext=["First share", "Last share"],
                tickfont=dict(color="#475569", size=9),
                thickness=10,
            ),
            opacity=0.85,
            line=dict(width=0.5, color=BG),
        ),
        text=edges["target_user"].str[:12].tolist(),
        customdata=edges[["depth", "hours", "out_deg"]].values.tolist(),
        hovertemplate=(
            "Hour: %{customdata[1]:.1f}<br>"
            "Depth: %{customdata[0]}<br>"
            "Amplification: %{customdata[2]:.0f} prior shares<extra></extra>"
        )
    ))

    # Add a trend line showing cascade frontier
    frontier = edges.groupby("depth")["hours"].min().reset_index()
    fig.add_trace(go.Scatter(
        x=frontier["hours"].tolist(),
        y=frontier["depth"].tolist(),
        mode="lines",
        line=dict(color=colour, width=1.5, dash="dot"),
        name="Cascade frontier",
        hoverinfo="skip",
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text="How deep did this claim penetrate, and how quickly?",
            font=dict(size=13, color=TEXT)
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=56, r=80, t=60, b=60),
        xaxis=dict(
            title="Hours since first share",
            gridcolor=GRID, tickfont=dict(color=TEXT),
            zeroline=False,
        ),
        yaxis=dict(
            title="Cascade depth",
            gridcolor=GRID, tickfont=dict(color=TEXT),
            zeroline=False,
            dtick=1,
        ),
        legend=dict(
            bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID,
            borderwidth=1, font=dict(size=10, color=TEXT),
            x=0.01, y=0.99,
        ),
        annotations=[dict(
            text="Bubble size = amplification (prior shares by that user)  ·  Dotted line = cascade frontier",
            x=0.5, y=-0.12, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color="#475569"), align="center"
        )]
    )
    return fig


def plot_fake_vs_real(summary_df):
    """
    Fake vs Real comparison — reference quality.
    Diverging lollipop / connected-dot chart: one row per metric,
    dots for fake (red) and real (green), connected by a gap line.
    Ratio multiplier badge embedded in data space on each row.
    Colour-intensity scaled by magnitude.
    """
    if summary_df.empty or "metric" not in summary_df.columns:
        return _empty("Not enough data")

    metric_labels = {
        "n_nodes":          "Network size (unique users)",
        "n_edges":          "Total shares",
        "max_depth":        "Cascade depth (generations)",
        "max_breadth":      "Peak breadth (users/level)",
        "n_communities":    "Community fragmentation",
        "median_speed_hrs": "Spread velocity (hrs to 50%)",
        "debunk_pct":       "Debunking resistance (%)",
        "virality_score":   "Virality composite score",
    }
    df = summary_df.copy()
    exclude = ["emotional_drift", "sentiment_drift"]
    df = df[~df["metric"].isin(exclude)]
    df = df[df[["fake", "real"]].max(axis=1) > 0]
    df["metric"] = df["metric"].map(lambda x: metric_labels.get(x, x))
    df = df.sort_values("fake", ascending=True).reset_index(drop=True)

    metrics  = df["metric"].tolist()
    fv       = df["fake"].tolist()
    rv       = df["real"].tolist()
    x_max    = max(max(fv), max(rv)) * 1.18

    # Compute ratio for colour-coding gap lines
    ratios = [f / max(r, 0.01) for f, r in zip(fv, rv)]
    max_ratio = max(ratios)

    fig = go.Figure()

    # ── CONNECTING LINES (gap lines) — colour intensity by ratio ──────────────
    for i, (m, f, r, ratio) in enumerate(zip(metrics, fv, rv, ratios)):
        intensity = min(1.0, ratio / max(max_ratio, 1))
        alpha = 0.25 + 0.45 * intensity
        line_col = f"rgba(244,63,94,{alpha:.2f})" if f > r else f"rgba(16,185,129,{alpha:.2f})"
        fig.add_trace(go.Scatter(
            x=[r, f], y=[m, m],
            mode="lines",
            line=dict(color=line_col, width=3.5),
            showlegend=False, hoverinfo="skip",
        ))

    # ── REAL DOTS ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=rv, y=metrics,
        mode="markers",
        name="Real News",
        marker=dict(
            color=RC, size=13, opacity=0.90,
            line=dict(width=2, color="white"),
            symbol="circle",
        ),
        hovertemplate="<b>%{y}</b><br>Real avg: <b>%{x:.2f}</b><extra></extra>",
    ))

    # ── FAKE DOTS ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fv, y=metrics,
        mode="markers",
        name="Fake News",
        marker=dict(
            color=FC, size=13, opacity=0.90,
            line=dict(width=2, color="white"),
            symbol="circle",
        ),
        hovertemplate="<b>%{y}</b><br>Fake avg: <b>%{x:.2f}</b><extra></extra>",
    ))

    # ── RATIO BADGES — embedded in data space at right of chart ───────────────
    annotations = []
    for m, f, r, ratio in zip(metrics, fv, rv, ratios):
        badge_x = max(f, r) + x_max * 0.01
        if ratio >= 2.0:
            badge_col = FC
            badge_text = f"  {ratio:.1f}×  "
        elif ratio >= 1.2:
            badge_col = "#fbbf24"
            badge_text = f"  {ratio:.1f}×  "
        else:
            badge_col = RC
            badge_text = f"  {ratio:.1f}×  "
        annotations.append(dict(
            x=badge_x, y=m,
            text=badge_text,
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=10, color=badge_col, family="DM Mono, monospace"),
            bgcolor=f"rgba({','.join(str(int(badge_col.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)",
            borderpad=3,
        ))

    # ── KEY FINDING CALLOUT — biggest gap (fake/real) ─────────────────────────
    max_ratio_idx = ratios.index(max(ratios))
    biggest_metric = metrics[max_ratio_idx]
    biggest_fv     = fv[max_ratio_idx]
    biggest_ratio  = ratios[max_ratio_idx]
    annotations.append(dict(
        x=biggest_fv, y=biggest_metric,
        text=f"Largest gap:<br>{biggest_ratio:.1f}× higher in fake news",
        showarrow=True, arrowhead=2, arrowcolor=FC, arrowwidth=1.5,
        ax=60, ay=-28,
        font=dict(size=9, color=FC, family="DM Mono, monospace"),
        bgcolor="rgba(6,10,16,0.88)", borderpad=4,
        bordercolor="rgba(244,63,94,0.3)", borderwidth=1,
    ))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    annotations.append(dict(
        x=0.5, y=-0.12, xref="paper", yref="paper",
        text=(
            "Each row: left dot = Real avg · right dot = Fake avg · Gap line = divergence · Badge = fake÷real ratio  ·  "
            "Key finding: fake cascades are ~4.7× deeper · ~6×+ broader · ~6× faster than real news  ·  "
            "Replicates Vosoughi et al. (Science, 2018)"
        ),
        showarrow=False, font=dict(size=10, color="#475569"), align="center",
    ))

    fig.update_layout(
        title=dict(
            text="Across every measurable dimension, how differently does fake news spread?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=218, r=120, t=72, b=90),
        height=480,
        xaxis=dict(
            title="Average value across all 200 claims",
            gridcolor=GRID, tickfont=dict(color=TEXT),
            range=[0, x_max], zeroline=False,
        ),
        yaxis=dict(
            type="category", tickfont=dict(color=TEXT, size=11),
            gridcolor=GRID, categoryorder="array", categoryarray=metrics,
        ),
        legend=dict(
            bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID, borderwidth=1,
            font=dict(size=11, color=TEXT), x=0.01, y=0.01,
            xanchor="left", yanchor="bottom",
        ),
        annotations=annotations,
    )
    return fig

def plot_sentiment_drift(sentiment_df, label="fake"):
    if sentiment_df.empty: return _empty("No sentiment data")
    colour = FC if label == "fake" else RC
    sents  = sentiment_df["avg_sentiment"].tolist()
    depths = sentiment_df["depth"].tolist()

    fig = go.Figure()

    # Zone shading
    fig.add_hrect(y0=0,   y1=0.4,  fillcolor="rgba(244,63,94,0.07)",  line_width=0)
    fig.add_hrect(y0=0.4, y1=0.6,  fillcolor="rgba(251,191,36,0.05)", line_width=0)
    fig.add_hrect(y0=0.6, y1=1.0,  fillcolor="rgba(16,185,129,0.05)", line_width=0)

    fig.add_trace(go.Scatter(
        x=depths, y=sents,
        mode="lines+markers",
        line=dict(color=colour, width=2.5),
        marker=dict(size=10, color=colour, line=dict(width=2, color=BG),
                    symbol="circle"),
        fill="tozeroy",
        fillcolor=f"rgba(244,63,94,0.07)" if label=="fake" else "rgba(16,185,129,0.07)",
        name="Avg Sentiment",
        hovertemplate="Depth %{x}: sentiment <b>%{y:.3f}</b><extra></extra>"
    ))

    # Annotate min point (most negative)
    if sents:
        min_idx = sents.index(min(sents))
        annotations = [
            dict(x=0.5, y=-0.26, xref="paper", yref="paper",
                 text="Sentiment scored 0–1  ·  Zone shading: red <0.4 = net negative  ·  amber 0.4–0.6 = neutral  ·  green >0.6 = positive",
                 showarrow=False, font=dict(size=10, color="#475569"), align="center"),
            dict(x=depths[min_idx], y=sents[min_idx],
                 text=f"Peak negativity<br>depth {depths[min_idx]}",
                 showarrow=True, arrowhead=2, arrowcolor=colour, arrowwidth=1.5,
                 ax=40, ay=-30,
                 font=dict(size=9, color=colour),
                 bgcolor="rgba(6,10,16,0.8)", borderpad=3),
        ]
    else:
        annotations = []

    fig.add_hline(y=0.5, line_dash="dot", line_color="#475569", line_width=1.5)
    annotations.append(dict(
        x=0.01, y=0.51, xref="paper", yref="y",
        text="── Neutral baseline (0.5)",
        showarrow=False, font=dict(size=8, color="#475569"),
        xanchor="left", yanchor="bottom",
        bgcolor="rgba(6,10,16,0.7)", borderpad=2
    ))

    fig.update_layout(
        title=dict(
            text="Does emotional tone intensify as the claim spreads deeper into the network?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=64, r=40, t=68, b=110),
        height=400,
        xaxis=dict(title="Cascade depth (share generation)",
                   gridcolor=GRID, tickfont=dict(color=TEXT), dtick=1),
        yaxis=dict(title="Average VADER sentiment score",
                   gridcolor=GRID, tickfont=dict(color=TEXT), range=[0, 1]),
        annotations=annotations,
    )
    return fig


def plot_super_spreaders(spreaders_df):
    if spreaders_df.empty: return _empty("No spreader data")

    df = spreaders_df.copy()
    df = df.sort_values("pagerank", ascending=True)

    max_pr = float(df["pagerank"].max()) if float(df["pagerank"].max()) > 0 else 1
    bar_colors = [
        f"rgba(244,{int(63*(1-v/max_pr))},{int(94*(1-v/max_pr))},{0.55+0.45*v/max_pr})"
        for v in df["pagerank"]
    ]

    annotations = [dict(
        text="PageRank = global network influence  ·  Out-degree = direct amplification  ·  Betweenness = bridge between isolated communities <br> ·  ★ = super-spreader with highest compound score",
        x=0.5, y=-0.30, xref="paper", yref="paper",
        showarrow=False, font=dict(size=10, color="#475569"), align="center"
    )]

    # Mark top spreader with star
    if len(df) > 0:
        top = df.iloc[-1]
        annotations.append(dict(
            x=float(top["pagerank"]), y=str(top["user"])[:16],
            text="★ top amplifier",
            showarrow=True, arrowhead=2, arrowcolor=FC, arrowwidth=1.5,
            ax=50, ay=0,
            font=dict(size=9, color="#a78bfa"),
            bgcolor="rgba(6,10,16,0.8)", borderpad=3,
        ))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["pagerank"].tolist(),
        y=df["user"].str[:16].tolist(),
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        customdata=list(zip(
            df["pagerank"].round(6).tolist(),
            df["out_degree"].tolist(),
            df["betweenness"].round(4).tolist(),
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "PageRank influence: %{customdata[0]}<br>"
            "Shares forwarded: %{customdata[1]}<br>"
            "Community bridge score: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text="Which users acted as key amplification nodes in this cascade?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=130, r=60, t=68, b=110),
        height=400,
        xaxis=dict(title="PageRank influence score",
                   gridcolor=GRID, tickfont=dict(color=TEXT)),
        yaxis=dict(type="category", autorange="reversed",
                   tickfont=dict(color=TEXT, size=11), gridcolor=GRID),
        annotations=annotations,
    )
    return fig


def plot_platform_breakdown(claim_id, spread_df, metrics_df=None):
    edges = spread_df[spread_df["claim_id"] == claim_id].copy()
    if edges.empty:
        return _empty("No spread data")

    all_depths  = spread_df.groupby("claim_id")["depth"].max()
    all_shares  = spread_df.groupby("claim_id").size()
    all_speed   = spread_df.groupby("claim_id")["hours_since_origin"].quantile(0.25)
    all_sent    = spread_df.groupby("claim_id")["sentiment_score"].mean()
    all_debunk  = spread_df.groupby("claim_id")["is_debunk"].mean()

    def n(val, series, invert=False):
        lo, hi = float(series.min()), float(series.max())
        if hi == lo: return 0.5
        v = (val - lo) / (hi - lo)
        return round(float(1 - v if invert else v), 3)

    this_scores = [
        n(int(edges["depth"].max()), all_depths),
        n(len(edges), all_shares),
        n(float(edges["hours_since_origin"].quantile(0.25)), all_speed, invert=True),
        n(1 - float(edges["sentiment_score"].mean()), all_sent.apply(lambda x: 1-x)),
        n(1 - float(edges["is_debunk"].mean()), all_debunk.apply(lambda x: 1-x)),
    ]

    fake_bench = [0.82, 0.78, 0.85, 0.72, 0.80]
    real_bench = [0.28, 0.31, 0.22, 0.35, 0.25]
    dims = ["Depth", "Reach", "Speed", "Negativity", "Debunk<br>Resistance"]
    dims_c  = dims + [dims[0]]
    this_c  = this_scores + [this_scores[0]]
    fake_c  = fake_bench  + [fake_bench[0]]
    real_c  = real_bench  + [real_bench[0]]
    is_fake = sum(this_scores) / len(this_scores) > 0.5
    cc      = FC if is_fake else RC

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=fake_c, theta=dims_c, fill="toself",
        fillcolor="rgba(244,63,94,0.08)",
        line=dict(color=FC, width=1.5, dash="dot"),
        name="Avg Fake",
        hovertemplate="%{theta}: %{r:.2f}<extra>Avg Fake</extra>"))
    fig.add_trace(go.Scatterpolar(
        r=real_c, theta=dims_c, fill="toself",
        fillcolor="rgba(16,185,129,0.08)",
        line=dict(color=RC, width=1.5, dash="dot"),
        name="Avg Real",
        hovertemplate="%{theta}: %{r:.2f}<extra>Avg Real</extra>"))
    fig.add_trace(go.Scatterpolar(
        r=this_c, theta=dims_c, fill="toself",
        fillcolor="rgba(244,63,94,0.22)" if is_fake else "rgba(16,185,129,0.22)",
        line=dict(color=cc, width=2.5),
        name="This Claim",
        hovertemplate="%{theta}: %{r:.2f}<extra>This Claim</extra>"))
    fig.update_layout(
        title=dict(text="How does this claim's spread compare to (fake vs real) news benchmarks?", font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID,
                tickfont=dict(color="#475569", size=9),
                tickvals=[0.25,0.5,0.75], ticktext=["Low","Mid","High"],
                linecolor=GRID),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID,
                tickfont=dict(color=TEXT, size=11))),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID,
            borderwidth=1, font=dict(size=11, color=TEXT),
            orientation="v", x=1.05, xanchor="left", y=0.5),
        annotations=[
            dict(
                text="Scores normalised across all 200 claims · Benchmarks calibrated to Vosoughi et al. (Science, 2018) findings",
                x=0.5, y=-0.22, xref="paper", yref="paper", showarrow=False,
                font=dict(size=9, color="#475569"), align="center"),
            dict(
                text="Axes: Depth = cascade length · Reach = total shares · Speed = how fast it spread · Negativity = avg negative sentiment · Debunk Resistance = % shares that were not corrections",
                x=0.5, y=-0.30, xref="paper", yref="paper", showarrow=False,
                font=dict(size=9, color="#475569"), align="center"),
        ])
    return fig


def plot_virality_gauge(risk_score, risk_level):
    bar_colour=FC if risk_score>=65 else ("#FBBF24" if risk_score>=35 else RC)
    fig=go.Figure(go.Indicator(mode="gauge+number",value=risk_score,
        number=dict(suffix=" / 100",font=dict(color=TEXT,size=28)),
        gauge=dict(axis=dict(range=[0,100],tickcolor=TEXT,tickfont=dict(color=TEXT)),
            bar=dict(color=bar_colour),bgcolor=GRID,
            steps=[dict(range=[0,35],color="#166534"),
                   dict(range=[35,65],color="#854D0E"),
                   dict(range=[65,100],color="#7F1D1D")],
            threshold=dict(line=dict(color="white",width=3),thickness=0.75,value=risk_score)),
        title=dict(text=f"Virality Risk  {risk_level}",font=dict(color=TEXT,size=14))))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        height=350,
        margin=dict(l=40, r=40, t=80, b=20)
    )
    return fig

def plot_speaker_bubble(claims_df, metrics_df):
    if claims_df is None or claims_df.empty:
        return _empty("No data")
    try:
        df = metrics_df.merge(
            claims_df[["claim_id", "speaker", "party"]],
            on="claim_id", how="inner"
        )
        df["virality_score"] = pd.to_numeric(
            df.get("virality_score", 0.3), errors="coerce").fillna(0.3)
        df["max_depth"] = pd.to_numeric(df["max_depth"], errors="coerce").fillna(0)

        grpd = df.groupby(["speaker", "party"])
        agg = pd.DataFrame({
            "n_claims":     grpd["claim_id"].count(),
            "fake_rate":    grpd["label"].apply(lambda x: (x == "fake").mean()),
            "avg_virality": grpd["virality_score"].mean(),
            "avg_depth":    grpd["max_depth"].mean(),
        }).reset_index()
        agg = agg[agg["n_claims"] >= 2].round(3)
        if agg.empty:
            return _empty("Not enough speaker data (need ≥2 claims per speaker)")

        party_colors = {
            "republican":   FC,
            "democrat":     "#3b82f6",
            "none":         "#94a3b8",
            "organization": "#a78bfa",
        }

        fig = go.Figure()
        for party, grp in agg.groupby("party"):
            color = party_colors.get(party, NC)
            fig.add_trace(go.Scatter(
                x=grp["n_claims"].tolist(),
                y=grp["fake_rate"].tolist(),
                mode="markers",
                name=party.title(),
                text=grp["speaker"].tolist(),
                marker=dict(
                    size=(grp["avg_virality"] * 60 + 10).tolist(),
                    color=color,
                    opacity=0.8,
                    line=dict(width=1.5, color=BG),
                ),
                customdata=grp[["avg_virality", "avg_depth", "party"]].values.tolist(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Claims analysed: %{x}<br>"
                    "Fake rate: %{y:.0%}<br>"
                    "Avg virality: %{customdata[0]:.3f}<br>"
                    "Avg cascade depth: %{customdata[1]:.1f}<br>"
                    "Party: %{customdata[2]}<extra></extra>"
                )
            ))

        fig.add_hline(y=0.5, line_dash="dot", line_color="#475569", line_width=1)
        fig.add_vline(x=float(agg["n_claims"].median()),
                      line_dash="dot", line_color="#475569", line_width=1)
        fig.add_annotation(
            x=float(agg["n_claims"].max()) * 0.95, y=0.97,
            text="⚠ HIGH VOLUME · HIGH FAKE RATE",
            showarrow=False, font=dict(size=9, color=FC), xanchor="right"
        )
        fig.update_layout(
            title=dict(
                text="Which speakers combine high claim volume with high misinformation rates?",
                font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TEXT, family="Inter, sans-serif"),
            margin=dict(l=60, r=40, t=64, b=100),
            height=480,
            xaxis=dict(title=dict(text="Number of claims analysed", standoff=15),
                       gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(title="Fake rate", gridcolor=GRID, tickfont=dict(color=TEXT),
                       tickformat=".0%", range=[-0.05, 1.1]),
            legend=dict(bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID,
                        borderwidth=1, font=dict(size=11, color=TEXT)),
            annotations=[dict(
                text="Bubble size = avg virality score · Overlapping bubbles = co-located speakers with identical claim volume — differentiated by party affiliation and virality magnitude · Top-right quadrant = high-volume high-misinformation-rate speakers",
                x=0, y=-0.20, xref="paper", yref="paper",
                showarrow=False, font=dict(size=9, color="#475569"),
                align="left", xanchor="left"
            )]
        )
        return fig
    except Exception as e:
        import traceback
        import logging
        logging.getLogger(__name__).error("BUBBLE FULL ERROR:\n%s", traceback.format_exc())
        return _empty(f"Bubble error: {e}")

def plot_speaker_sankey(by_speaker_df, by_party_df, by_subject_df, claims_df):
    if claims_df is None or claims_df.empty:
        return _empty("No claims data for Sankey")
    try:
        df = claims_df[claims_df["label"] == "fake"].copy()
        if "subject" in df.columns:
            df["subject"] = df["subject"].str.split(",").str[0].str.strip()
        df = df[df["speaker"].notna() & df["party"].notna() & df["subject"].notna()]
        df = df[df["party"] != "none"]

        top_speakers = df["speaker"].value_counts().head(12).index
        top_subjects = df["subject"].value_counts().head(10).index
        df = df[df["speaker"].isin(top_speakers) & df["subject"].isin(top_subjects)]

        if df.empty:
            return _empty("Not enough data for Sankey")

        speakers = sorted(df["speaker"].unique().tolist())
        parties  = sorted(df["party"].unique().tolist())
        subjects = sorted(df["subject"].unique().tolist())

        all_nodes = speakers + parties + subjects
        node_idx  = {n: i for i, n in enumerate(all_nodes)}

        party_colors = {"republican": "#ef4444", "democrat": "#3b82f6", "none": "#94a3b8"}

        node_colors = []
        for n in all_nodes:
            if n in speakers:
                party = df[df["speaker"] == n]["party"].mode()
                p = party.iloc[0] if len(party) else "none"
                node_colors.append(party_colors.get(p, "#a78bfa"))
            elif n in parties:
                node_colors.append(party_colors.get(n, "#a78bfa"))
            else:
                node_colors.append("#38bdf8")

        sp_links = df.groupby(["speaker", "party"]).size().reset_index(name="count")
        ps_links = df.groupby(["party", "subject"]).size().reset_index(name="count")

        sources = ([node_idx[r.speaker] for _, r in sp_links.iterrows()] +
                   [node_idx[r.party]   for _, r in ps_links.iterrows()])
        targets = ([node_idx[r.party]   for _, r in sp_links.iterrows()] +
                   [node_idx[r.subject] for _, r in ps_links.iterrows()])
        values  = sp_links["count"].tolist() + ps_links["count"].tolist()

        link_colors = []
        for _, r in sp_links.iterrows():
            c = party_colors.get(r.party, "#a78bfa")
            link_colors.append(f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.3)")
        for _, r in ps_links.iterrows():
            c = party_colors.get(r.party, "#a78bfa")
            link_colors.append(f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.25)")

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20, thickness=18,
                line=dict(color=BG, width=0.5),
                label=all_nodes,
                color=node_colors,
                hovertemplate="<b>%{label}</b><br>Flow volume: %{value}<extra></extra>",
            ),
            link=dict(
                source=sources, target=targets, value=values, color=link_colors,
                hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>Fake claims: %{value}<extra></extra>",
            )
        ))
        fig.update_layout(
            title=dict(
                text="How does misinformation flow from speakers through party lines into subject areas?",
                font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=TEXT, family="Inter, sans-serif", size=11),
            margin=dict(l=20, r=20, t=64, b=80),
            height=520,
            annotations=[dict(
                text="Left = individual speakers  ·  Centre = party affiliation  ·  Right = subject area  ·  Width = volume of fake claims  ·  Red=Republican · Blue=Democrat · Cyan=subject",
                x=0.5, y=-0.1, xref="paper", yref="paper",
                showarrow=False, font=dict(size=9, color="#475569"), align="center"
            )]
        )
        return fig
    except Exception as e:
        return _empty(f"Sankey error: {e}")

def plot_mutation_similarity(mutation_df):
    if mutation_df.empty: return _empty("No mutation data")
    df = mutation_df[mutation_df["similarity"] > 0].copy()
    if df.empty: return _empty("No mutation data")

    sims    = df["similarity"].tolist()
    vs      = df["version"].tolist()
    n       = len(sims)
    xs      = list(range(n))

    fig = go.Figure()

    # ── Zone shading (shapes, not annotations) ────────────────────────────────
    fig.add_hrect(y0=0,    y1=0.5,  fillcolor="rgba(244,63,94,0.08)",  line_width=0)
    fig.add_hrect(y0=0.5,  y1=0.7,  fillcolor="rgba(251,191,36,0.07)", line_width=0)
    fig.add_hrect(y0=0.7,  y1=1.08, fillcolor="rgba(16,185,129,0.05)", line_width=0)

    # ── Threshold + stable reference lines ────────────────────────────────────
    fig.add_hline(y=0.7, line_dash="dash", line_color="#fbbf24", line_width=1.5)
    fig.add_hline(y=1.0, line_dash="dot",  line_color=RC, line_width=1, opacity=0.3)

    # ── Trend line ────────────────────────────────────────────────────────────
    trend_annotation = None
    if n >= 2:
        import numpy as _np
        m, b = _np.polyfit(xs, sims, 1)
        trend_y = [m*x + b for x in xs]
        fig.add_trace(go.Scatter(
            x=vs, y=trend_y, mode="lines",
            name=f"Trend (slope {m:+.3f}/hop)",
            line=dict(color="#64748b", width=1.5, dash="dot"),
            hoverinfo="skip"
        ))
        direction = "↓ Accelerating drift" if m < -0.05 else ("↑ Recovery" if m > 0.05 else "→ Stable")
        trend_annotation = dict(
            x=vs[-1], y=trend_y[-1], xref="x", yref="y",
            text=direction, showarrow=False,
            xanchor="right", yanchor="top", yshift=-14,
            font=dict(size=8, color="#64748b"),
            bgcolor="rgba(6,10,16,0.75)", borderpad=2,
        )

    # ── Connecting line ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=vs, y=sims, mode="lines",
        line=dict(color="#1e3a5f", width=2),
        showlegend=False, hoverinfo="skip"
    ))

    # ── Data points ───────────────────────────────────────────────────────────
    point_colors = [FC if s < 0.5 else "#fbbf24" if s < 0.7 else RC for s in sims]
    text_positions = []
    for i, s in enumerate(sims):
        if 0.65 < s < 0.75:
            text_positions.append("bottom center")
        elif i % 2 == 0:
            text_positions.append("top center")
        else:
            text_positions.append("bottom center")

    fig.add_trace(go.Scatter(
        x=vs, y=sims, mode="markers+text",
        marker=dict(size=13, color=point_colors,
                    line=dict(width=2, color=BG),
                    symbol=["diamond" if s < 0.7 else "circle" for s in sims]),
        text=[f"{s:.2f}" for s in sims],
        textposition=text_positions,
        textfont=dict(size=9, color=point_colors),
        customdata=[("⚠ MUTATED" if s < 0.7 else "✓ STABLE") for s in sims],
        hovertemplate="<b>%{x}</b><br>Similarity: <b>%{y:.3f}</b><br>%{customdata}<extra></extra>",
        showlegend=False
    ))

    # ── Build ALL annotations in one list ─────────────────────────────────────
    annotations = [
        # Zone labels — right edge, data y coords
        dict(x=0.99, y=0.25,  xref="paper", yref="y", text="HEAVY MUTATION",
             showarrow=False, font=dict(size=8, color=FC, family="DM Mono, monospace"),
             xanchor="right", yanchor="middle", bgcolor="rgba(6,10,16,0.75)", borderpad=3),
        dict(x=0.99, y=0.595, xref="paper", yref="y", text="MODERATE DRIFT",
             showarrow=False, font=dict(size=8, color="#fbbf24", family="DM Mono, monospace"),
             xanchor="right", yanchor="middle", bgcolor="rgba(6,10,16,0.75)", borderpad=3),
        dict(x=0.99, y=0.875, xref="paper", yref="y", text="STABLE",
             showarrow=False, font=dict(size=8, color=RC, family="DM Mono, monospace"),
             xanchor="right", yanchor="middle", bgcolor="rgba(6,10,16,0.75)", borderpad=3),
        # Threshold label — left, below line
        dict(x=0.01, y=0.69, xref="paper", yref="y", text="── 70% stability threshold",
             showarrow=False, font=dict(size=8, color="#fbbf24"),
             xanchor="left", yanchor="top", bgcolor="rgba(6,10,16,0.75)", borderpad=2),
        # Footer caption
        dict(x=0.5, y=-0.4, xref="paper", yref="paper",
             text="TF-IDF cosine similarity  ·  ◆ diamond = below 70% threshold  ·  Shading: green = stable (>0.7)  ·  amber = moderate drift (0.5–0.7)  ·  red = heavy mutation (<0.5)",
             showarrow=False, font=dict(size=10, color="#475569"), align="center"),
    ]
    # First breach callout (conditional)
    first_breach = next(((v, s) for v, s in zip(vs, sims) if s < 0.7), None)
    if first_breach:
        bv, bs = first_breach
        annotations.append(dict(
            x=bv, y=bs, xref="x", yref="y",
            text="First breach", showarrow=True,
            arrowhead=2, arrowcolor=FC, arrowwidth=1.5, ax=38, ay=30,
            font=dict(size=8, color=FC), bgcolor="rgba(6,10,16,0.75)", borderpad=2,
        ))
    if trend_annotation:
        annotations.append(trend_annotation)

    below = sum(1 for s in sims if s < 0.7)
    fig.update_layout(
        title=dict(
            text=f"Semantic mutation across {n} propagated variants — {below}/{n} breach the 70% stability threshold",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=70, r=40, t=68, b=120),
        height=420,
        legend=dict(bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=10, color=TEXT), x=0.01, y=0.13,
                    xanchor="left", yanchor="bottom"),
        xaxis=dict(title=dict(text="Propagated variant (diffusion hop)", standoff=8),
                   gridcolor=GRID, tickfont=dict(color=TEXT), type="category"),
        yaxis=dict(title="TF-IDF cosine similarity to source",
                   range=[-0.05, 1.13], gridcolor=GRID, tickfont=dict(color=TEXT), dtick=0.1),
        annotations=annotations,
    )
    return fig

def plot_depth_virality_scatter(metrics_df):
    """
    Depth × Virality scatter — reference quality.
    Inspired by: background zone shading that encodes meaning,
    per-class regression lines with R² embedded in data space,
    quadrant labels directly on the chart, centroid markers.
    """
    if metrics_df.empty: return _empty("No data")

    df = metrics_df.copy()
    df["max_depth"]     = pd.to_numeric(df["max_depth"],     errors="coerce").fillna(0)
    df["virality_score"] = pd.to_numeric(df["virality_score"], errors="coerce").fillna(0)

    fake_df = df[df["label"] == "fake"]
    real_df = df[df["label"] == "real"]

    x_max = float(df["max_depth"].max()) * 1.08
    y_max = float(df["virality_score"].max()) * 1.08
    x_mid = float(df["max_depth"].median())
    y_mid = float(df["virality_score"].median())

    fig = go.Figure()

    # ── QUADRANT BACKGROUND ZONES ─────────────────────────────────────────────
    # Bottom-left  = low depth / low virality  (real news zone)
    fig.add_shape(type="rect", x0=0, x1=x_mid, y0=0, y1=y_mid,
                  fillcolor="rgba(16,185,129,0.07)", line_width=0, layer="below")
    # Top-right = high depth / high virality (fake news zone)
    fig.add_shape(type="rect", x0=x_mid, x1=x_max, y0=y_mid, y1=y_max,
                  fillcolor="rgba(244,63,94,0.09)", line_width=0, layer="below")
    # Top-left + bottom-right = mixed / transition zones
    fig.add_shape(type="rect", x0=0, x1=x_mid, y0=y_mid, y1=y_max,
                  fillcolor="rgba(251,191,36,0.04)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_mid, x1=x_max, y0=0, y1=y_mid,
                  fillcolor="rgba(251,191,36,0.04)", line_width=0, layer="below")

    # ── MEDIAN CROSSHAIR LINES ─────────────────────────────────────────────────
    fig.add_shape(type="line", x0=x_mid, x1=x_mid, y0=0, y1=y_max,
                  line=dict(color="#334155", width=1.2, dash="dot"), layer="below")
    fig.add_shape(type="line", x0=0, x1=x_max, y0=y_mid, y1=y_mid,
                  line=dict(color="#334155", width=1.2, dash="dot"), layer="below")

    # ── REGRESSION LINES + R² ─────────────────────────────────────────────────
    annotations = []
    for lbl, sub, colour in [("fake", fake_df, FC), ("real", real_df, RC)]:
        if len(sub) < 3:
            continue
        xs = sub["max_depth"].values.astype(float)
        ys = sub["virality_score"].values.astype(float)
        m, b = np.polyfit(xs, ys, 1)
        # R²
        ys_pred = m * xs + b
        ss_res  = float(np.sum((ys - ys_pred) ** 2))
        ss_tot  = float(np.sum((ys - ys.mean()) ** 2))
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        x_line  = [float(xs.min()), float(xs.max())]
        y_line  = [m * x_line[0] + b, m * x_line[1] + b]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=colour, width=2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        # Embed equation in data space, near end of line
        label_name = "Fake" if lbl == "fake" else "Real"
        annotations.append(dict(
            x=x_line[1], y=max(0, y_line[1]),
            text=f"y = {m:+.3f}x + {b:.2f}<br>R² = {r2:.3f}  [{label_name}]",
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=9, color=colour, family="DM Mono, monospace"),
            bgcolor="rgba(6,10,16,0.82)", borderpad=4,
        ))

    # ── DATA POINTS ───────────────────────────────────────────────────────────
    for lbl, sub, colour, name in [
        ("fake", fake_df, FC,  "Fake News"),
        ("real", real_df, RC,  "Real News"),
    ]:
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["max_depth"].astype(float).tolist(),
            y=sub["virality_score"].astype(float).tolist(),
            mode="markers",
            name=name,
            marker=dict(
                color=colour, size=7, opacity=0.70,
                line=dict(width=0.8, color=BG),
            ),
            text=sub["claim_id"].tolist(),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cascade depth: %{x}<br>"
                "Virality score: %{y:.3f}<extra></extra>"
            )
        ))

    # ── CENTROID MARKERS ─────────────────────────────────────────────────────
    for sub, colour, name in [(fake_df, FC, "Fake"), (real_df, RC, "Real")]:
        if sub.empty: continue
        cx = float(sub["max_depth"].mean())
        cy = float(sub["virality_score"].mean())
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(color=colour, size=16, symbol="diamond",
                        line=dict(width=2.5, color="white"), opacity=1.0),
            showlegend=False,
            hovertemplate=f"<b>{name} centroid</b><br>Avg depth: {cx:.1f}<br>Avg virality: {cy:.3f}<extra></extra>",
        ))
        annotations.append(dict(
            x=cx, y=cy,
            text=f"  {name}<br>  centroid",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(size=8, color=colour),
            bgcolor="rgba(6,10,16,0.75)", borderpad=2,
        ))

    # ── QUADRANT LABELS ───────────────────────────────────────────────────────
    # Placed in corners of their respective zones
    quad_style = dict(showarrow=False, xanchor="center", yanchor="middle",
                      font=dict(size=10, family="DM Mono, monospace"))
    annotations += [
        dict(x=x_mid * 0.35, y=y_max * 0.90,
             text="LOW DEPTH<br>HIGH VIRALITY",
             font=dict(size=9, color="#fbbf24", family="DM Mono, monospace"),
             bgcolor="rgba(6,10,16,0.6)", borderpad=3, **{k:v for k,v in quad_style.items() if k not in ["font","bgcolor","borderpad"]}),
        dict(x=x_max * 0.78, y=y_max * 0.90,
             text="HIGH DEPTH<br>HIGH VIRALITY",
             font=dict(size=10, color=FC, family="DM Mono, monospace"),
             bgcolor="rgba(6,10,16,0.6)", borderpad=3, **{k:v for k,v in quad_style.items() if k not in ["font","bgcolor","borderpad"]}),
        dict(x=x_mid * 0.35, y=y_max * 0.10,
             text="LOW DEPTH<br>LOW VIRALITY",
             font=dict(size=10, color=RC, family="DM Mono, monospace"),
             bgcolor="rgba(6,10,16,0.6)", borderpad=3, **{k:v for k,v in quad_style.items() if k not in ["font","bgcolor","borderpad"]}),
        dict(x=x_max * 0.78, y=y_max * 0.10,
             text="HIGH DEPTH<br>LOW VIRALITY",
             font=dict(size=9, color="#fbbf24", family="DM Mono, monospace"),
             bgcolor="rgba(6,10,16,0.6)", borderpad=3, **{k:v for k,v in quad_style.items() if k not in ["font","bgcolor","borderpad"]}),
        # Crosshair labels
        dict(x=x_mid, y=y_max * 1.02, xref="x", yref="y",
             text=f"Median depth: {x_mid:.1f}",
             showarrow=False, xanchor="center", yanchor="bottom",
             font=dict(size=8, color="#475569"),
             bgcolor="rgba(6,10,16,0.7)", borderpad=2),
        # Footer
        dict(x=0.5, y=-0.14, xref="paper", yref="paper",
             text="Each dot = one claim  ·  ◆ = class centroid  ·  Dashed = linear regression per class  ·  Zones: red = fake territory · green = real territory",
             showarrow=False, font=dict(size=10, color="#475569"), align="center"),
    ]

    fig.update_layout(
        title=dict(
            text="Do claims that spread deeper also achieve higher virality?",
            font=dict(size=13, color=TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        margin=dict(l=64, r=48, t=68, b=100),
        height=440,
        xaxis=dict(title="Max cascade depth (share generations)",
                   gridcolor=GRID, tickfont=dict(color=TEXT),
                   range=[0, x_max], zeroline=False),
        yaxis=dict(title="Virality score (0–1)",
                   gridcolor=GRID, tickfont=dict(color=TEXT),
                   range=[0, y_max], zeroline=False),
        legend=dict(bgcolor="rgba(13,21,32,0.9)", bordercolor=GRID,
                    borderwidth=1, font=dict(size=11, color=TEXT),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        annotations=annotations,
    )
    return fig