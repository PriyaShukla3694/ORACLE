"""Simulation engine running decision-support episodes and strategy comparisons.

Evaluates observation trajectories, realized emergency response costs, and regret.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from oracle_logic import (
    BaseStrategy,
    BeliefState,
    FixedPriority,
    HighestUncertainty,
    OracleLookahead2,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
)
from routing import FAIL_PENALTY, best_route, enumerate_routes, get_route_edges
from scenario_data import SOURCES, RoadEdge, Scenario, get_roads_list, sample_worlds


def run_episode(
    scenario: Scenario,
    strategy: BaseStrategy,
    budget: float = 3.0,
    budget_mode: str = "cost",
    source: str = "perfect",
    n_worlds: int = 2000,
    length_scale: float = 2.0,
    seed: int = 42,
    worlds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Execute an observation episode under a budget and return evaluation metrics.

    Args:
        scenario: Scenario instance containing graph and hidden ground truth.
        strategy: Observation ranking strategy to test.
        budget: Total observation budget available (cost or observation count).
        budget_mode: "cost" (costs subtracted from budget) or "count" (k observations allowed).
        source: Observation source ('perfect', 'field', 'satellite').
        n_worlds: Number of simulated worlds for Monte Carlo belief state.
        length_scale: Spatial correlation length scale.
        seed: Random seed for world sampling and strategy randomness.
        worlds: Optional pre-sampled worlds array of shape (S, E).

    Returns:
        dict containing observation log, chosen route, realized cost, omniscient cost, regret.
    """
    rng = np.random.default_rng(seed)
    graph = scenario.graph
    roads = get_roads_list(graph)
    T, A = enumerate_routes(graph)
    R = len(T)
    if R == 0:
        raise ValueError("Network has no valid routes from BASE to HOSPITAL.")

    # Sample prior worlds or use pre-sampled
    if worlds is not None:
        W = np.array(worlds, copy=False)
    else:
        W = sample_worlds(graph, n_worlds=n_worlds, length_scale=length_scale, rng=rng)
    weights = np.ones(W.shape[0], dtype=float) / float(W.shape[0])

    belief = BeliefState(
        worlds=W,
        weights=weights,
        edge_metadata=roads,
        graph=graph,
        T=T,
        A=A,
    )

    _, initial_v0 = best_route(belief.cost, belief.weights)

    remaining_budget = float(budget)
    observation_log: List[Dict[str, Any]] = []
    step_num = 0

    while remaining_budget > 0:
        candidates = [r for r in roads if r.road_id not in belief.observed_roads]
        if not candidates:
            break

        try:
            ranked = strategy.rank(belief, candidates, remaining_budget=remaining_budget)
        except TypeError:
            ranked = strategy.rank(belief, candidates)
        # Select highest-ranked candidate within remaining budget
        chosen_candidate = None
        chosen_score = 0.0
        chosen_details: Dict[str, Any] = {}

        for rid, score, details in ranked:
            edge_obj = belief.edge_metadata[belief.road_id_to_idx[rid]]
            deduct_amount = 1.0 if budget_mode == "count" else edge_obj.observation_cost
            if deduct_amount <= remaining_budget:
                chosen_candidate = edge_obj
                chosen_score = score
                chosen_details = details
                break

        if chosen_candidate is None:
            # No candidate fits in remaining budget
            break

        step_num += 1
        rid = chosen_candidate.road_id
        deduct_amount = 1.0 if budget_mode == "count" else chosen_candidate.observation_cost
        obs_cost = chosen_candidate.observation_cost
        _, v0_before = best_route(belief.cost, belief.weights)

        # Query scenario observation
        reported = scenario.observe(rid, source=source)
        remaining_budget -= deduct_amount

        # Update belief
        accuracy = SOURCES.get(source, 1.0)
        belief.update(rid, reported, accuracy=accuracy)
        _, v0_after = best_route(belief.cost, belief.weights)

        observation_log.append(
            {
                "step": step_num,
                "road_id": rid,
                "road_name": chosen_candidate.road_name,
                "cost": obs_cost,
                "reported": reported,
                "score": round(float(chosen_score), 4),
                "expected_cost_before": round(v0_before, 2),
                "expected_cost_after": round(v0_after, 2),
                "remaining_budget": round(remaining_budget, 2),
                "details": chosen_details,
            }
        )

    # Agent's final chosen route based on posterior belief
    final_r_idx, final_exp_cost = best_route(belief.cost, belief.weights)
    final_route_edges = get_route_edges(A, roads, final_r_idx)
    final_route_rids = [e.road_id for e in final_route_edges]
    final_route_time = float(T[final_r_idx])

    # Evaluate realized cost from private ground truth
    # Route fails if any edge on it is blocked in truth
    is_route_blocked = any(not scenario._truth[rid] for rid in final_route_rids)
    realized_cost = final_route_time + (FAIL_PENALTY if is_route_blocked else 0.0)

    # Evaluate omniscient best route with full ground truth knowledge
    omniscient_best_idx = 0
    omniscient_cost = float("inf")
    for r in range(R):
        r_edges = get_route_edges(A, roads, r)
        r_time = float(T[r])
        blocked = any(not scenario._truth[e.road_id] for e in r_edges)
        r_cost = r_time + (FAIL_PENALTY if blocked else 0.0)
        if r_cost < omniscient_cost:
            omniscient_cost = r_cost
            omniscient_best_idx = r

    omniscient_edges = get_route_edges(A, roads, omniscient_best_idx)
    regret = max(0.0, realized_cost - omniscient_cost)

    return {
        "strategy": strategy.name,
        "budget": budget,
        "remaining_budget": round(remaining_budget, 2),
        "budget_spent": round(budget - remaining_budget, 2),
        "num_observations": len(observation_log),
        "observation_log": observation_log,
        "initial_expected_cost": round(initial_v0, 2),
        "final_expected_cost": round(final_exp_cost, 2),
        "final_route_idx": final_r_idx,
        "final_route_roads": final_route_rids,
        "final_route_travel_time": round(final_route_time, 2),
        "is_final_route_blocked": is_route_blocked,
        "realized_cost": round(realized_cost, 2),
        "omniscient_route_idx": omniscient_best_idx,
        "omniscient_route_roads": [e.road_id for e in omniscient_edges],
        "omniscient_cost": round(omniscient_cost, 2),
        "regret": round(regret, 2),
    }


def compare_strategies(
    scenario_factory: Callable[[], Scenario],
    strategies: Optional[List[BaseStrategy]] = None,
    budget: float = 3.0,
    source: str = "perfect",
    n_worlds: int = 2000,
    length_scale: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare multiple observation strategies on an identical scenario under identical budget.

    Returns a pandas DataFrame summarizing observation decisions and realized costs.
    """
    if strategies is None:
        strategies = [
            FixedPriority(),
            HighestUncertainty(),
            RandomStrategy(rng=np.random.default_rng(seed)),
            WeightedHeuristic(),
            OracleVOI(),
        ]

    # Create base scenario to extract graph and ground truth
    base_scenario = scenario_factory()
    graph = base_scenario.graph
    fixed_truth = dict(base_scenario._truth)

    results: List[Dict[str, Any]] = []

    for strat in strategies:
        # Clone scenario with identical ground truth and new observation tracker
        sim_scenario = Scenario(
            graph=graph,
            truth=dict(fixed_truth),
            rng=np.random.default_rng(seed),
        )

        res = run_episode(
            scenario=sim_scenario,
            strategy=strat,
            budget=budget,
            source=source,
            n_worlds=n_worlds,
            length_scale=length_scale,
            seed=seed,
        )

        obs_rids = [entry["road_id"] for entry in res["observation_log"]]
        results.append(
            {
                "Strategy": res["strategy"],
                "Observed Roads": ", ".join(obs_rids) if obs_rids else "None",
                "Budget Spent": res["budget_spent"],
                "Chosen Route": f"Route #{res['final_route_idx']} ({'BLOCKED' if res['is_final_route_blocked'] else 'OPEN'})",
                "Expected Cost": res["final_expected_cost"],
                "Realized Cost": res["realized_cost"],
                "Omniscient Cost": res["omniscient_cost"],
                "Regret": res["regret"],
            }
        )

    return pd.DataFrame(results)


def benchmark_random_scenarios(
    n_scenarios: int = 15,
    budget: float = 4.0,
    source: str = "perfect",
    n_worlds: int = 1500,
    seed_start: int = 100,
) -> pd.DataFrame:
    """Benchmark strategies across multiple synthetic flood scenarios to separate VOI from heuristics."""
    from scenario_data import random_scenario

    strategies = [
        FixedPriority(),
        HighestUncertainty(),
        RandomStrategy(rng=np.random.default_rng(42)),
        WeightedHeuristic(),
        OracleVOI(),
    ]

    metrics = {s.name: {"regrets": [], "costs": [], "zero_regret_count": 0} for s in strategies}

    for i in range(n_scenarios):
        scen_seed = seed_start + i
        base_scen = random_scenario(seed=scen_seed, n_nodes=9)
        fixed_truth = dict(base_scen._truth)

        for s in strategies:
            sim = Scenario(base_scen.graph, dict(fixed_truth), np.random.default_rng(scen_seed))
            res = run_episode(sim, s, budget=budget, source=source, n_worlds=n_worlds, seed=scen_seed)
            metrics[s.name]["regrets"].append(res["regret"])
            metrics[s.name]["costs"].append(res["realized_cost"])
            if res["regret"] <= 1e-4:
                metrics[s.name]["zero_regret_count"] += 1

    summary_rows = []
    for s in strategies:
        regrets = metrics[s.name]["regrets"]
        costs = metrics[s.name]["costs"]
        zero_pct = (metrics[s.name]["zero_regret_count"] / n_scenarios) * 100.0
        summary_rows.append(
            {
                "Strategy": s.name,
                "Mean Regret (min)": round(float(np.mean(regrets)), 2),
                "Regret Std": round(float(np.std(regrets)), 2),
                "Mean Realized Cost (min)": round(float(np.mean(costs)), 2),
                "Zero-Regret Rate (%)": f"{zero_pct:.1f}%",
            }
        )

    return pd.DataFrame(summary_rows)
