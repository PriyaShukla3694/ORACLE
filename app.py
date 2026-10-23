"""ORACLE: Value-of-Information Decision Engine for Flood Emergency Routing.

Streamlit Interactive Decision-Support Dashboard.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oracle_logic import (
    BeliefState,
    CorridorClearanceBaseFirst,
    CorridorFirstHospitalEnd,
    HighestUncertainty,
    OracleLookahead2,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
    explain,
    voi,
)
from routing import FAIL_PENALTY, best_route, enumerate_routes, get_route_edges
from scenario_data import (
    SOURCES,
    RoadEdge,
    Scenario,
    demo_scenario,
    get_roads_list,
    random_scenario,
    sample_worlds,
)
from simulate import run_episode

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="ORACLE | Flood Emergency Routing Decision Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Theme (#08111F navy, #35D6EA cyan, #F2B84B amber, #2BBFA8 teal)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #08111F;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .disclaimer-banner {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        border-left: 4px solid #F2B84B;
        padding: 10px 16px;
        border-radius: 4px;
        color: #F8FAFC;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #0F1D32;
        border: 1px solid #1E2E4A;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.25);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #35D6EA;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 2px;
    }
    .rec-card {
        background: linear-gradient(135deg, #132744 0%, #0D1C33 100%);
        border: 1px solid #F2B84B;
        border-radius: 8px;
        padding: 18px 22px;
        box-shadow: 0 0 15px rgba(242, 184, 75, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B1728;
        padding: 6px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2E4A !important;
        color: #35D6EA !important;
    }
    </style>
    <div class="disclaimer-banner">
        ⚠️ <strong>Decision-support simulation — not an operational emergency system. All data simulated.</strong>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Data Loaders
# ==============================================================================

@st.cache_data
def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load precomputed benchmark CSV tables with friendly error handling."""
    path = Path(__file__).resolve().parent / filepath
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ==============================================================================
# Helper Functions: Network Plotting & Coloring
# ==============================================================================

def get_color_for_p(p: float) -> str:
    """Map passability probability p in [0, 1] to continuous RGB hex (Red -> Amber -> Green)."""
    p = max(0.0, min(1.0, float(p)))
    if p < 0.5:
        # Red (239, 68, 68) to Amber (245, 158, 11)
        ratio = p / 0.5
        r = int(239 + (245 - 239) * ratio)
        g = int(68 + (158 - 68) * ratio)
        b = int(68 + (11 - 68) * ratio)
    else:
        # Amber (245, 158, 11) to Green (16, 185, 129)
        ratio = (p - 0.5) / 0.5
        r = int(245 + (16 - 245) * ratio)
        g = int(158 + (185 - 158) * ratio)
        b = int(11 + (129 - 11) * ratio)
    return f"rgb({r},{g},{b})"


def plot_network(
    belief: BeliefState,
    best_route_idx: int,
    recommended_road_id: Optional[str] = None,
    show_ground_truth: bool = False,
    scenario: Optional[Scenario] = None,
) -> go.Figure:
    """Render interactive Plotly network with node coordinates and passability styling."""
    graph = belief.graph
    roads = belief.edge_metadata
    marginals = belief.marginals()
    best_edges = get_route_edges(belief.A, roads, best_route_idx) if best_route_idx >= 0 else []
    best_rids = {e.road_id for e in best_edges}

    fig = go.Figure()

    # 1. Draw Edges
    for r in roads:
        e_idx = belief.road_id_to_idx[r.road_id]
        p = float(marginals[e_idx])
        pos_u = graph.nodes[r.u]["pos"]
        pos_v = graph.nodes[r.v]["pos"]
        mid_x = (pos_u[0] + pos_v[0]) / 2.0
        mid_y = (pos_u[1] + pos_v[1]) / 2.0

        is_observed = r.road_id in belief.observed_roads
        is_on_best = r.road_id in best_rids
        is_recommended = (r.road_id == recommended_road_id)

        # Base edge line
        edge_color = get_color_for_p(p)
        width = 3.5

        if is_on_best:
            # Highlight current best route in Cyan
            fig.add_trace(
                go.Scatter(
                    x=[pos_u[0], pos_v[0]],
                    y=[pos_u[1], pos_v[1]],
                    mode="lines",
                    line=dict(color="#35D6EA", width=8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if is_recommended:
            # Highlight recommended road in Amber glow
            fig.add_trace(
                go.Scatter(
                    x=[pos_u[0], pos_v[0]],
                    y=[pos_u[1], pos_v[1]],
                    mode="lines",
                    line=dict(color="#F2B84B", width=6, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        hover_text = (
            f"<b>{r.road_name} ({r.road_id})</b><br>"
            f"Current P(Passable): {p*100:.1f}%<br>"
            f"Travel Time: {r.travel_time:.1f} min<br>"
            f"Observation Cost: {r.observation_cost:.1f}<br>"
            f"Staleness: {r.staleness:.2f}<br>"
            f"Status: {belief.observed_roads.get(r.road_id, 'UNOBSERVED').upper()}"
        )
        if show_ground_truth and scenario is not None:
            true_st = "OPEN" if scenario._truth.get(r.road_id, False) else "BLOCKED"
            hover_text += f"<br><b>Ground Truth: {true_st}</b>"

        fig.add_trace(
            go.Scatter(
                x=[pos_u[0], pos_v[0]],
                y=[pos_u[1], pos_v[1]],
                mode="lines",
                line=dict(color=edge_color, width=width),
                hoverinfo="text",
                text=hover_text,
                showlegend=False,
            )
        )

        # Midpoint status markers for observed edges
        if is_observed:
            rep = belief.observed_roads[r.road_id]
            rep_color = "#10B981" if rep == "passable" else "#EF4444"
            rep_symbol = "circle"
            rep_label = "PASSABLE" if rep == "passable" else "BLOCKED"

            fig.add_trace(
                go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode="markers+text",
                    marker=dict(size=12, color=rep_color, line=dict(width=2, color="#FFFFFF")),
                    text=[f"{rep_label}"],
                    textposition="top center",
                    textfont=dict(size=10, color=rep_color, family="monospace"),
                    hoverinfo="text",
                    hovertext=f"Observed: {r.road_id} -> {rep.upper()}",
                    showlegend=False,
                )
            )

    # 2. Draw Nodes
    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []

    for n, data in graph.nodes(data=True):
        pos = data["pos"]
        node_x.append(pos[0])
        node_y.append(pos[1])
        if n == "BASE":
            node_color.append("#2BBFA8")
            node_size.append(20)
            node_symbol.append("diamond")
            node_text.append("<b>BASE (Depot)</b>")
        elif n == "HOSPITAL":
            node_color.append("#35D6EA")
            node_size.append(22)
            node_symbol.append("cross")
            node_text.append("<b>HOSPITAL</b>")
        else:
            node_color.append("#94A3B8")
            node_size.append(14)
            node_symbol.append("circle")
            node_text.append(f"Node {n}")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_size,
                color=node_color,
                symbol=node_symbol,
                line=dict(width=2, color="#08111F"),
            ),
            text=node_text,
            textposition="bottom center",
            textfont=dict(color="#F8FAFC", size=11),
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Invisible traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#35D6EA", width=6), name="Current Best Route"))
    if recommended_road_id:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#F2B84B", width=4, dash="dash"), name="Recommended Observation"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#10B981", size=10), name="High P(Passable)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="#EF4444", size=10), name="Low P(Passable)"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A1628",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#94A3B8", size=10),
            bgcolor="rgba(15, 29, 50, 0.7)",
        ),
        hovermode="closest",
        height=520,
    )

    return fig


# ==============================================================================
# Session State Management
# ==============================================================================

def initialize_session_state(force_reset: bool = False) -> None:
    """Initialize or reset session state variables."""
    if force_reset or "scenario" not in st.session_state:
        st.session_state.clear()
        st.session_state.initialized = True
        st.session_state.observation_log = []
        st.session_state.finalized = False
        st.session_state.final_results = None
        st.session_state.force_truth_mode = "Sampled from prior"


# ==============================================================================
# Strategy Factory
# ==============================================================================

STRATEGY_OPTIONS = {
    "ORACLE-VOI": lambda acc, b_mode: OracleVOI(accuracy=acc, budget_mode=b_mode),
    "ORACLE-Lookahead-2 (Exploratory: advantage over single-step VOI not established in mean regret)": lambda acc, b_mode: OracleLookahead2(accuracy=acc, budget_mode=b_mode),
    "Weighted-Heuristic": lambda acc, b_mode: WeightedHeuristic(budget_mode=b_mode),
    "Corridor-first (hospital->base)": lambda acc, b_mode: CorridorFirstHospitalEnd(),
    "Corridor-clearance (base->hospital)": lambda acc, b_mode: CorridorClearanceBaseFirst(),
    "Highest-Uncertainty": lambda acc, b_mode: HighestUncertainty(),
    "Random": lambda acc, b_mode: RandomStrategy(rng=np.random.default_rng(42)),
}


# ==============================================================================
# Sidebar Configuration
# ==============================================================================

initialize_session_state()

with st.sidebar:
    st.title("🌊 ORACLE Controls")
    st.markdown("### Decision Engine Settings")

    # Scenario Selection
    scen_type = st.radio(
        "Scenario Type",
        options=["Illustrative Demo Scenario", "Synthetic Random Scenario"],
        index=0,
    )

    seed_val = 42
    setting_val = "mixed"
    length_scale_val = 2.0
    n_nodes_val = 8

    if scen_type == "Illustrative Demo Scenario":
        st.info("6-node illustrative network with a fast primary route and safer detour.")
        demo_truth_choice = st.selectbox(
            "Force Hidden Truth for Demo:",
            options=[
                "Sampled from prior",
                "Force Road B PASSABLE (Unlocks 19m primary route)",
                "Force Road B BLOCKED (Corridor disaster, +120m penalty)",
            ],
            index=0,
        )
    else:
        seed_val = st.number_input("Random Topology Seed", min_value=0, max_value=999999, value=42, step=1)
        setting_val = st.selectbox(
            "Environmental Flooding Setting",
            options=["mixed", "reliable", "harsh"],
            index=0,
            help="'harsh' denotes severe disruption where only ~17% of scenarios have any passable route.",
        )
        if setting_val == "harsh":
            st.caption("ℹ️ *Harsh setting: only ~17% of scenarios have any passable route.*")
        length_scale_val = st.slider("Spatial Correlation Length Scale (ℓ)", 0.0, 5.0, 2.0, 0.5, help="0.0 = independent roads.")
        n_nodes_val = st.slider("Network Nodes", 6, 12, 8, 1)

    st.markdown("---")
    st.markdown("### Policy & Observation Parameters")

    # Strategy Selection
    selected_strategy_key = st.selectbox(
        "Observation Strategy",
        options=list(STRATEGY_OPTIONS.keys()),
        index=0,
    )

    # Observation Source
    source_choice = st.selectbox(
        "Observation Sensor Source",
        options=["perfect", "satellite", "field"],
        index=0,
        help="Satellite operates with asymmetric classification (TPR 0.88, TNR 0.82; mean 0.85).",
    )
    if source_choice == "satellite":
        st.caption("🛰️ *Asymmetric sensor: TPR=0.88 (passable), TNR=0.82 (blocked).*")

    # Sample Size
    sample_size_S = st.slider("Monte Carlo Sample Size (S)", 500, 5000, 2000, 500)

    # Budget
    budget_mode = st.radio("Budget Mode", options=["count", "cost"], index=0)
    if budget_mode == "count":
        budget_limit = st.selectbox("Max Observations (k)", options=[1.0, 2.0, 3.0, 5.0, 7.0], index=2)
    else:
        budget_limit = st.slider("Total Cost Budget", 2.0, 20.0, 6.0, 1.0)

    st.markdown("---")
    if st.button("🔄 Reset Scenario & Belief", use_container_width=True):
        initialize_session_state(force_reset=True)
        st.rerun()


# ==============================================================================
# Model Initialization & Dynamic State Cache
# ==============================================================================

# Check if scenario parameters changed
config_signature = (
    scen_type,
    seed_val,
    setting_val,
    length_scale_val,
    n_nodes_val,
    source_choice,
    sample_size_S,
    budget_mode,
    budget_limit,
    demo_truth_choice if scen_type == "Illustrative Demo Scenario" else None,
)

if st.session_state.get("last_config") != config_signature or "belief" not in st.session_state:
    st.session_state.last_config = config_signature
    st.session_state.observation_log = []
    st.session_state.finalized = False
    st.session_state.final_results = None

    # Instantiate Scenario
    if scen_type == "Illustrative Demo Scenario":
        scen = demo_scenario(seed=seed_val, length_scale=length_scale_val)
        if demo_truth_choice == "Force Road B PASSABLE (Unlocks 19m primary route)":
            scen._truth["ROAD_B"] = True
            scen._truth["ROAD_BASE_MAIN"] = True
            scen._truth["ROAD_A"] = True
        elif demo_truth_choice == "Force Road B BLOCKED (Corridor disaster, +120m penalty)":
            scen._truth["ROAD_B"] = False
    else:
        scen = random_scenario(seed=seed_val, n_nodes=n_nodes_val, length_scale=length_scale_val, setting=setting_val)

    st.session_state.scenario = scen
    graph = scen.graph
    roads = get_roads_list(graph)
    T, A = enumerate_routes(graph)

    # Sample prior worlds
    rng = np.random.default_rng(seed_val)
    W = sample_worlds(graph, n_worlds=sample_size_S, length_scale=length_scale_val, rng=rng)
    weights = np.ones(sample_size_S, dtype=float) / float(sample_size_S)

    st.session_state.belief = BeliefState(W, weights, roads, graph, T, A)
    st.session_state.remaining_budget = float(budget_limit)

scenario: Scenario = st.session_state.scenario
belief: BeliefState = st.session_state.belief
roads: List[RoadEdge] = belief.edge_metadata
sensor_acc = SOURCES.get(source_choice, 1.0)

# Instantiate selected strategy
active_strategy = STRATEGY_OPTIONS[selected_strategy_key](sensor_acc, budget_mode)

# Candidates remaining
unobserved_candidates = [r for r in roads if r.road_id not in belief.observed_roads]

# Measure recommendation latency
rec_t0 = time.perf_counter()
if unobserved_candidates and st.session_state.remaining_budget > 0 and not st.session_state.finalized:
    try:
        ranked_candidates = active_strategy.rank(belief, unobserved_candidates, remaining_budget=st.session_state.remaining_budget)
    except Exception:
        ranked_candidates = active_strategy.rank(belief, unobserved_candidates)
else:
    ranked_candidates = []
rec_t1 = time.perf_counter()
rec_latency_ms = (rec_t1 - rec_t0) * 1000.0

current_best_route_idx, current_v0 = best_route(belief.cost, belief.weights)
top_recommendation_id = ranked_candidates[0][0] if ranked_candidates else None


# ==============================================================================
# TABS
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Live Decision",
    "⚖️ Compare Strategies",
    "📊 Experiment Results",
    "📖 About & Limits",
])


# ==============================================================================
# TAB 1: Live Decision
# ==============================================================================

with tab1:
    # Status Banner & Top Metrics
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Expected Cost (V₀)</div>
                <div class="metric-value">{current_v0:.1f} m</div>
                <div class="metric-sub">Route #{current_best_route_idx} (T={belief.T[current_best_route_idx]:.1f}m)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Remaining Budget</div>
                <div class="metric-value">{st.session_state.remaining_budget:.1f}</div>
                <div class="metric-sub">{budget_mode.upper()} mode</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Queries Made</div>
                <div class="metric-value">{len(st.session_state.observation_log)}</div>
                <div class="metric-sub">{len(belief.observed_roads)}/{len(roads)} roads checked</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m4:
        rec_display = top_recommendation_id if top_recommendation_id else "None (Done)"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Top Recommended Road</div>
                <div class="metric-value" style="color:#F2B84B;">{rec_display}</div>
                <div class="metric-sub">Latency: {rec_latency_ms:.1f} ms</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m5:
        status_txt = "✅ Route Finalized" if st.session_state.finalized else "🔍 Observing"
        status_color = "#2BBFA8" if st.session_state.finalized else "#35D6EA"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-title">Decision State</div>
                <div class="metric-value" style="color:{status_color}; font-size:1.25rem;">{status_txt}</div>
                <div class="metric-sub">Strategy: {active_strategy.name[:18]}...</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Layout: Network Visualization (Left 65%) vs Decision Actions & Card (Right 35%)
    col_net, col_actions = st.columns([65, 35])

    with col_net:
        st.markdown("#### 🗺️ Network Map & Passability Belief")
        fig_net = plot_network(
            belief=belief,
            best_route_idx=current_best_route_idx,
            recommended_road_id=top_recommendation_id if not st.session_state.finalized else None,
            show_ground_truth=st.session_state.finalized,
            scenario=scenario,
        )
        st.plotly_chart(fig_net, use_container_width=True)

        if st.session_state.finalized:
            st.success("🎉 Decision finalized! Ground truth passability is now revealed on the map.")

    with col_actions:
        st.markdown("#### 🎯 Recommendation & Actions")

        if not st.session_state.finalized and top_recommendation_id:
            # Show Recommendation Explanation Card
            explanation_text = explain(belief, top_recommendation_id, accuracy=sensor_acc)
            top_rec_edge = belief.edge_metadata[belief.road_id_to_idx[top_recommendation_id]]
            top_rec_score = ranked_candidates[0][1]

            st.markdown(
                f"""<div class="rec-card">
                    <div style="font-size:0.85rem; color:#F2B84B; font-weight:700; text-transform:uppercase;">
                        ORACLE Top Recommendation
                    </div>
                    <div style="font-size:1.35rem; color:#FFFFFF; font-weight:700; margin: 4px 0;">
                        {top_rec_edge.road_name} ({top_recommendation_id})
                    </div>
                    <div style="font-size:0.88rem; color:#CBD5E1; line-height:1.5;">
                        Prior P(Passable): <b>{top_rec_edge.p_passable*100:.1f}%</b> | Cost: <b>{top_rec_edge.observation_cost:.1f}</b> | Priority Score: <b>{top_rec_score:.3f}</b>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander("📝 Detailed Policy Explanation", expanded=True):
                st.text(explanation_text)

            st.markdown("---")

            # Observation Action Buttons
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                deduct_amt = 1.0 if budget_mode == "count" else top_rec_edge.observation_cost
                btn_disabled = (deduct_amt > st.session_state.remaining_budget)

                if st.button(
                    f"🔍 Observe {top_recommendation_id}",
                    type="primary",
                    disabled=btn_disabled,
                    use_container_width=True,
                    help=f"Deducts {deduct_amt:.1f} from remaining budget.",
                ):
                    try:
                        rep = scenario.observe(top_recommendation_id, source=source_choice)
                        st.session_state.remaining_budget -= deduct_amt
                        belief.update(top_recommendation_id, rep, accuracy=sensor_acc)
                        st.session_state.observation_log.append({
                            "step": len(st.session_state.observation_log) + 1,
                            "road_id": top_recommendation_id,
                            "reported": rep,
                            "cost": top_rec_edge.observation_cost,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Observation error: {e}")

            with c_btn2:
                # Custom road observation
                manual_road = st.selectbox(
                    "Or choose road:",
                    options=[r.road_id for r in unobserved_candidates],
                    label_visibility="collapsed",
                )
                manual_edge = belief.edge_metadata[belief.road_id_to_idx[manual_road]]
                m_deduct = 1.0 if budget_mode == "count" else manual_edge.observation_cost

                if st.button("Observe Custom", disabled=(m_deduct > st.session_state.remaining_budget), use_container_width=True):
                    try:
                        rep = scenario.observe(manual_road, source=source_choice)
                        st.session_state.remaining_budget -= m_deduct
                        belief.update(manual_road, rep, accuracy=sensor_acc)
                        st.session_state.observation_log.append({
                            "step": len(st.session_state.observation_log) + 1,
                            "road_id": manual_road,
                            "reported": rep,
                            "cost": manual_edge.observation_cost,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Observation error: {e}")

        # Finalize Decision Button
        if not st.session_state.finalized:
            if st.button("🏁 Finalize Route Decision & Commit Dispatch", type="secondary", use_container_width=True):
                st.session_state.finalized = True

                # Evaluate final outcome against private ground truth
                final_r, final_exp = best_route(belief.cost, belief.weights)
                final_edges = get_route_edges(belief.A, roads, final_r)
                final_rids = [e.road_id for e in final_edges]
                is_blocked = any(not scenario._truth[r] for r in final_rids)
                realized = float(belief.T[final_r]) + (FAIL_PENALTY if is_blocked else 0.0)

                # Omniscient best route
                R_total = len(belief.T)
                omniscient_cost = float("inf")
                omniscient_r = 0
                for r_idx in range(R_total):
                    r_rids = [e.road_id for e in get_route_edges(belief.A, roads, r_idx)]
                    blk = any(not scenario._truth[r] for r in r_rids)
                    c = float(belief.T[r_idx]) + (FAIL_PENALTY if blk else 0.0)
                    if c < omniscient_cost:
                        omniscient_cost = c
                        omniscient_r = r_idx

                regret = max(0.0, realized - omniscient_cost)

                st.session_state.final_results = {
                    "final_route": final_r,
                    "final_route_roads": final_rids,
                    "final_travel_time": belief.T[final_r],
                    "is_blocked": is_blocked,
                    "realized_cost": realized,
                    "omniscient_cost": omniscient_cost,
                    "omniscient_route": omniscient_r,
                    "regret": regret,
                }
                st.rerun()

        # If Finalized, Display Reveal Card
        if st.session_state.finalized and st.session_state.final_results:
            res = st.session_state.final_results
            if res["is_blocked"]:
                st.error(f"""
                🚨 **CATASTROPHIC FAILURE: Route #{res['final_route']} was BLOCKED!**  
                Realized Journey Cost: **{res['realized_cost']:.1f} min** (Nominal: {res['final_travel_time']:.1f}m + **120.0m failure penalty**).  
                Omniscient Best Cost: **{res['omniscient_cost']:.1f} min**. Regret: **{res['regret']:.1f} min**.
                """)
            else:
                st.success(f"""
                🎉 **SAFE EMERGENCY TRANSIT: Route #{res['final_route']} is FULLY PASSABLE!**  
                Realized Journey Cost: **{res['realized_cost']:.1f} min** (0 failure penalty).  
                Omniscient Best Cost: **{res['omniscient_cost']:.1f} min**. Regret: **{res['regret']:.1f} min**.
                """)

            st.markdown("##### Route Ground Truth Status:")
            for rid in res["final_route_roads"]:
                is_open = scenario._truth[rid]
                obs_status = belief.observed_roads.get(rid, "Unobserved")
                st.write(f"- `{rid}`: {'🟢 OPEN' if is_open else '🔴 BLOCKED'} (Observation: *{obs_status}*)")

            if st.button("Start New Run", use_container_width=True):
                initialize_session_state(force_reset=True)
                st.rerun()

    # Candidate Ranking Table
    st.markdown("---")
    st.markdown("#### 📋 Candidate Roads Ranking Table")

    candidate_table_rows = []
    marginals = belief.marginals()
    for r in roads:
        e_idx = belief.road_id_to_idx[r.road_id]
        p_val = float(marginals[e_idx])
        is_obs = r.road_id in belief.observed_roads
        rep_val = belief.observed_roads.get(r.road_id, "—")

        # Find score and rank if in ranked_candidates
        rank_pos = "—"
        score_val = "—"
        voi_val = "—"
        for idx, (c_rid, c_sc, c_det) in enumerate(ranked_candidates, 1):
            if c_rid == r.road_id:
                rank_pos = f"#{idx}"
                score_val = f"{c_sc:.3f}"
                voi_val = f"{c_det.get('voi', c_det.get('joint_voi', 0.0)):.2f}"
                break

        on_best = "⭐ Current Best" if r.road_id in best_rids else ("Bypass / Alt" if np.sum(belief.A[:, e_idx]) > 0 else "None (Spur)")

        candidate_table_rows.append({
            "Road ID": r.road_id,
            "Road Name": r.road_name,
            "P(Passable)": f"{p_val*100:.1f}%",
            "Obs Cost": f"{r.observation_cost:.1f}",
            "Staleness": f"{r.staleness:.2f}",
            "VOI (min)": voi_val,
            "Priority Score": score_val,
            "Rank": rank_pos,
            "Route Incidence": on_best,
            "Report": rep_val.upper(),
        })

    df_cand = pd.DataFrame(candidate_table_rows)
    st.dataframe(df_cand, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 2: Compare Strategies
# ==============================================================================

with tab2:
    st.markdown("### ⚖️ Multi-Strategy Fair Benchmark Comparison")
    st.markdown(
        "Compare all 7 observation strategies on the **exact same scenario topology, same prior, same ground truth, and identical budget**."
    )

    if st.button("⚡ Run Multi-Strategy Comparison on Current Scenario", type="primary"):
        with st.spinner("Executing simulation episodes across all 7 strategies..."):
            comp_strategies = [
                OracleVOI(accuracy=sensor_acc, budget_mode=budget_mode),
                OracleLookahead2(accuracy=sensor_acc, budget_mode=budget_mode),
                WeightedHeuristic(budget_mode=budget_mode),
                CorridorFirstHospitalEnd(),
                CorridorClearanceBaseFirst(),
                HighestUncertainty(),
                RandomStrategy(rng=np.random.default_rng(seed_val)),
            ]

            comp_results = []
            comp_trajectories = {}

            # Freeze ground truth
            fixed_truth = dict(scenario._truth)

            for strat in comp_strategies:
                # Clone scenario with identical ground truth
                sim_scenario = Scenario(graph=scenario.graph, truth=dict(fixed_truth), rng=np.random.default_rng(seed_val))
                r_ep = run_episode(
                    scenario=sim_scenario,
                    strategy=strat,
                    budget=float(budget_limit),
                    budget_mode=budget_mode,
                    source=source_choice,
                    n_worlds=sample_size_S,
                    length_scale=length_scale_val,
                    seed=seed_val,
                )

                obs_seq = [s["road_id"] for s in r_ep["observation_log"]]
                comp_trajectories[strat.name] = r_ep["observation_log"]

                comp_results.append({
                    "Strategy": strat.name,
                    "Observed Roads": ", ".join(obs_seq) if obs_seq else "None",
                    "Budget Spent": r_ep["budget_spent"],
                    "Chosen Route": f"Route #{r_ep['final_route_idx']}",
                    "Route Status": "🔴 BLOCKED" if r_ep["is_final_route_blocked"] else "🟢 SAFE",
                    "Realized Cost (min)": r_ep["realized_cost"],
                    "Omniscient Cost (min)": r_ep["omniscient_cost"],
                    "Regret (min)": r_ep["regret"],
                })

            df_comp = pd.DataFrame(comp_results)
            st.session_state.comp_table = df_comp
            st.session_state.comp_trajectories = comp_trajectories

    if "comp_table" in st.session_state:
        df_comp = st.session_state.comp_table
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Spotlight Difference: ORACLE vs Highest-Uncertainty
        st.markdown("---")
        st.markdown("#### 🔍 Spotlight Analysis: ORACLE-VOI vs Highest-Uncertainty")

        oracle_row = df_comp[df_comp["Strategy"] == "ORACLE-VOI"].iloc[0]
        hu_row = df_comp[df_comp["Strategy"] == "Highest-Uncertainty"].iloc[0]

        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">ORACLE-VOI Regret</div>
                    <div class="metric-value" style="color:#2BBFA8;">{oracle_row['Regret (min)']:.1f} min</div>
                    <div class="metric-sub">Observed: {oracle_row['Observed Roads']} ({oracle_row['Route Status']})</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_sp2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Highest-Uncertainty Regret</div>
                    <div class="metric-value" style="color:#EF4444;">{hu_row['Regret (min)']:.1f} min</div>
                    <div class="metric-sub">Observed: {hu_row['Observed Roads']} ({hu_row['Route Status']})</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.caption(
            "Highest-Uncertainty greedily queries roads with maximum variance p(1-p), frequently wasting limited budget on unnavigable decoys. "
            "In contrast, ORACLE evaluates the decision tree over all candidate paths, querying only roads capable of altering route choice."
        )

        with st.expander("🔎 View Step-by-Step Observation Sequences"):
            for s_name, traj in st.session_state.comp_trajectories.items():
                st.markdown(f"**{s_name}:**")
                if not traj:
                    st.write("  *No observations made.*")
                for s in traj:
                    st.write(f"  - Step {s['step']}: Observed `{s['road_id']}` (Cost: {s['cost']:.1f}) -> Reported **{s['reported'].upper()}** | Score: {s['score']:.3f}")


# ==============================================================================
# TAB 3: Experiment Results
# ==============================================================================

with tab3:
    st.markdown("### 📊 Factorial Benchmark Results")
    st.markdown(
        "Comprehensive empirical results across **33,600 simulation episodes** (100 seeds × 3 settings × 2 length scales × 2 sources × 4 count budgets × 7 strategies). "
        "Confidence intervals (95%) computed via **cluster bootstrap clustered by scenario seed** ($B=1000$)."
    )

    df_summary = load_csv_data("results/headline_summary.csv")
    df_paired = load_csv_data("results/headline_paired.csv")

    if df_summary is None or df_paired is None:
        st.error(
            "Benchmark data files (`results/headline_summary.csv` or `results/headline_paired.csv`) not found. "
            "Please run `python evaluate_headline.py` to generate the benchmark tables."
        )
    else:
        # 1. Headline Metrics: Solvable-Only Scenarios
        st.markdown("#### 🎯 Headline Evaluation: Solvable-Only Scenarios (Count Budgets $k=1, 2, 3, 5$)")
        st.caption(
            "Filtered strictly to episodes where at least one passable route exists in ground truth (omniscient cost < 120 min). "
            "**Primary Metrics: Catastrophic-Failure Rate** (route truly blocked) and **Safe-Route Rate**."
        )

        df_solvable_summary = df_summary[df_summary["Subset"].str.contains("Solvable", case=False, na=False)].copy()
        st.dataframe(df_solvable_summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 2. Confirmatory Paired Differences
        st.markdown("#### ⚖️ Confirmatory Headline Paired Comparisons: ORACLE vs Baselines")
        st.caption(
            "Paired differences (Target minus Baseline) across all count-budget episodes with 95% cluster bootstrap CIs and Holm-adjusted p-values. "
            "Negative differences indicate lower failure rates / lower regret."
        )

        df_confirmatory = df_paired[df_paired["Analysis_Type"] == "Confirmatory Headline"].copy()
        st.dataframe(df_confirmatory, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 3. Diagnostic Cross-Tabulation
        st.markdown("#### 🔬 Empirical Diagnosis: Corridor-First vs ORACLE Zero-Regret Trade-Off")
        st.caption(
            "In solvable scenarios, Corridor-first achieves slightly higher exact-zero-regret rates (39.7% vs 33.3% / 37.7%) by blindly committing to the shortest corridor. "
            "However, this static gamble produces an **8.5% to 8.9% higher catastrophic failure rate** (30.5% vs 22.0% / 21.6%), resulting in substantially worse mean regret (37.15m vs 27.26m, p = 1.4×10⁻⁶)."
        )

        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown(
                """**Cross-Tabulation of Zero-Regret Outcomes (Solvable Scenarios, N=2,688):**
                - **Both Zero Regret:** 654 (24.3%)
                - **Only Corridor-First Zero:** 413 (15.4%)
                - **Only ORACLE-VOI Zero:** 241 (9.0%)
                - **Neither Zero Regret:** 1,380 (51.3%)  
                *Prior best route was already optimal in only 22.0% of solvable cases.*
                """
            )
        with col_diag2:
            st.markdown(
                """**Catastrophic Failure Rates by Route Switching Behavior:**
                - **Corridor-first (hospital→base):** Overall CF Rate = **30.5%**  
                  - When kept prior route: **29.4%** failure rate
                  - When changed route: **31.8%** failure rate
                - **ORACLE-VOI:** Overall CF Rate = **22.0%**  
                  - When kept prior route: **18.4%** failure rate
                  - When changed route: **26.0%** failure rate
                """
            )

        st.markdown("---")

        # 4. All-Scenarios Reference Table
        st.markdown("#### 🌐 All-Scenarios Reference Benchmark (Including Unnavigable Topologies)")
        df_all_summary = df_summary[df_summary["Subset"].str.contains("All Scenarios", case=False, na=False)].copy()
        st.dataframe(df_all_summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 5. Exploratory Subgroups
        st.markdown("#### 🧪 Exploratory Subgroup Results (Holm Multiplicity-Corrected)")
        st.caption(
            "All subgroup hypotheses are labeled as **Exploratory** with family-wise error rate controlled via the Holm step-down method."
        )

        subgroup_filter = st.selectbox(
            "Select Subgroup Dimension:",
            options=["Setting", "Budget_k", "Source", "Length_Scale"],
            index=0,
        )

        df_sub = df_paired[
            (df_paired["Analysis_Type"] == "Exploratory Subgroup") &
            (df_paired["Subgroup_Type"] == subgroup_filter)
        ].copy()

        st.dataframe(df_sub, use_container_width=True, hide_index=True)

        if subgroup_filter == "Setting":
            st.info("ℹ️ **Environmental Definition:** 'Harsh' setting strictly means only ~17% of scenarios have any passable route from BASE to HOSPITAL (omniscient passable rate = 17%).")


# ==============================================================================
# TAB 4: About & Limits
# ==============================================================================

with tab4:
    st.markdown("### 📖 About ORACLE, Theoretical Formulations & Boundaries")

    col_ab1, col_ab2 = st.columns(2)

    with col_ab1:
        st.markdown("#### 🧮 Mathematical Formulations")
        st.markdown(
            r"""
            **1. Value of Information (VOI):**
            $$\text{VOI}(e) = V_0 - \sum_{y \in \{P, B\}} P(y) \min_{r} \mathbb{E}[\text{Cost}(r) \mid y]$$
            where $V_0$ is the current baseline expected cost, and $y$ denotes the hypothetical observation outcome.

            **2. Emergency Journey Cost & Turnaround Penalty:**
            $$\text{Cost}(s, r) = \sum_{e \in r} T_e + \text{FAIL\_PENALTY} \times \mathbf{1}[\exists e \in r \text{ s.t. } W_{s,e} = \text{Blocked}]$$
            where $\text{FAIL\_PENALTY} = 120.0 \text{ min}$ models turnaround delay and forced rerouting.

            **3. Bayesian Likelihood Updating:**
            $$w_{s} \propto w_{s,0} \times P(\text{Report } y \mid W_{s,e})$$
            under sensor accuracy $P(y = \text{passable} \mid \text{passable}) = \text{TPR}$, $P(y = \text{blocked} \mid \text{blocked}) = \text{TNR}$.

            **4. Gaussian Copula Spatial Prior:**
            $$\Sigma_{ij} = \exp\left(-\frac{d_{ij}}{\ell}\right) + \epsilon I, \quad u_i = \Phi(z_i) \le p_i$$
            models correlated flood water levels across spatial road midpoints.
            """
        )

    with col_ab2:
        st.markdown("#### ⚠️ Technical Limitations & Decision Boundaries")
        st.markdown(
            """
            1. **Synthetic Data Boundaries:**
               - All networks, elevation midpoints, flood events, and sensor reports are **strictly simulated and synthetic**.
               - **No real-world empirical flood validation has been conducted.**

            2. **Residual Incomplete-Information Failures:**
               - Observation budgets are constrained. In solvable scenarios under average budgets (~2.75 queries), **unobserved links still cause catastrophic failure in ~22.0% of episodes**.

            3. **Severe Flooding Regime Limits:**
               - In the **Harsh** flooding setting (where only ~17% of scenarios possess any passable route), heuristic importance weighting on surviving paths achieves near parity with myopic VOI.

            4. **Role of System:**
               - ORACLE is an illustrative decision-support ideation model designed to inform human emergency coordinators, **not replace human command**.
               - **No claims of operational real-time dispatch readiness or lives saved are made.**
            """
        )

    st.markdown("---")
    st.markdown("##### 📦 System Architecture & Engine Details")
    st.write(
        "- **Stack:** Python 3.10+, NumPy (vectorized matrix operations), NetworkX (simple path enumeration), SciPy, Pandas, Plotly, Streamlit.\n"
        "- **Engine Tag:** `v0.1-engine`.\n"
        "- **Zero Ground-Truth Leakage:** BeliefState and strategies operate exclusively on Monte Carlo samples and prior weights without private `_truth` access."
    )
