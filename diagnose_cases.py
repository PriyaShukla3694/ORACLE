"""Diagnosis script replaying specific outlier cases from quick_eval step-by-step.

Compares:
- ORACLE-VOI (myopic single-step)
- ORACLE-Lookahead-2 (2-step joint VOI lookahead)

Cases:
1. Seed 1, length_scale 2.0, k=3, source perfect
2. Seed 2, length_scale 2.0, k=3 (and budget 3), source satellite
3. Seed 0, length_scale 2.0, k=5, source perfect
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from oracle_logic import BeliefState, OracleLookahead2, OracleVOI
from routing import FAIL_PENALTY, best_route, enumerate_routes, get_route_edges
from scenario_data import SOURCES, get_roads_list, random_scenario, sample_worlds


def replay_single_strategy(
    seed: int,
    length_scale: float,
    budget: float,
    budget_mode: str,
    source: str,
    strategy_cls: Any,
    strategy_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    scen = random_scenario(seed=seed, length_scale=length_scale)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    R = len(T)

    rng = np.random.default_rng(seed)
    W = sample_worlds(scen.graph, n_worlds=3000, length_scale=length_scale, rng=rng)
    weights = np.ones(3000, dtype=float) / 3000.0
    belief = BeliefState(W, weights, roads, scen.graph, T, A)

    accuracy = SOURCES.get(source, 1.0)
    strat = strategy_cls(accuracy=accuracy, budget_mode=budget_mode, **strategy_kwargs)

    remaining_budget = float(budget)
    steps_log: List[Dict[str, Any]] = []

    while remaining_budget > 0:
        candidates = [r for r in roads if r.road_id not in belief.observed_roads]
        if not candidates:
            break

        ranked = strat.rank(belief, candidates, remaining_budget=remaining_budget)
        chosen = None
        for rid, score, details in ranked:
            edge_obj = belief.edge_metadata[belief.road_id_to_idx[rid]]
            deduct = 1.0 if budget_mode == "count" else edge_obj.observation_cost
            if deduct <= remaining_budget:
                chosen = (edge_obj, score, details)
                break

        if chosen is None:
            break

        edge_obj, score, details = chosen
        rid = edge_obj.road_id
        deduct = 1.0 if budget_mode == "count" else edge_obj.observation_cost

        # Query scenario observation
        reported = scen.observe(rid, source=source)
        remaining_budget -= deduct

        # Belief update
        belief.update(rid, reported, accuracy=accuracy)
        new_best_r, new_v0 = best_route(belief.cost, belief.weights)

        steps_log.append({
            "road_id": rid,
            "cost": edge_obj.observation_cost,
            "score": score,
            "details": details,
            "reported": reported,
            "true": scen._truth[rid],
            "new_best_route": new_best_r,
            "new_v0": new_v0,
        })

    final_r, final_exp = best_route(belief.cost, belief.weights)
    final_edges = get_route_edges(A, roads, final_r)
    final_rids = [e.road_id for e in final_edges]
    is_final_blocked = any(not scen._truth[r] for r in final_rids)
    final_realized = T[final_r] + (FAIL_PENALTY if is_final_blocked else 0.0)

    # Omniscient best
    omniscient_cost = float("inf")
    omniscient_r = 0
    for r_idx in range(R):
        r_rids = [e.road_id for e in get_route_edges(A, roads, r_idx)]
        blocked = any(not scen._truth[r] for r in r_rids)
        c = T[r_idx] + (FAIL_PENALTY if blocked else 0.0)
        if c < omniscient_cost:
            omniscient_cost = c
            omniscient_r = r_idx

    regret = final_realized - omniscient_cost

    return {
        "strategy_name": strat.name,
        "steps_log": steps_log,
        "final_route": final_r,
        "final_route_roads": final_rids,
        "is_final_blocked": is_final_blocked,
        "realized_cost": final_realized,
        "omniscient_cost": omniscient_cost,
        "omniscient_route": omniscient_r,
        "regret": regret,
    }


def replay_case(seed: int, length_scale: float, budget: float, budget_mode: str, source: str) -> None:
    print("=" * 88)
    print(f"CASE DIAGNOSIS: Seed={seed}, LengthScale={length_scale}, Budget={budget} ({budget_mode}), Source={source}")
    print("=" * 88)

    res_voi = replay_single_strategy(
        seed=seed,
        length_scale=length_scale,
        budget=budget,
        budget_mode=budget_mode,
        source=source,
        strategy_cls=OracleVOI,
        strategy_kwargs={},
    )

    res_la2 = replay_single_strategy(
        seed=seed,
        length_scale=length_scale,
        budget=budget,
        budget_mode=budget_mode,
        source=source,
        strategy_cls=OracleLookahead2,
        strategy_kwargs={},
    )

    for res in (res_voi, res_la2):
        print(f"\n--- STRATEGY: {res['strategy_name']} ---")
        for i, step in enumerate(res["steps_log"], 1):
            det = step["details"]
            mode = det.get("mode", "single_voi")
            partner = det.get("best_partner", "N/A")
            j_voi = det.get("joint_voi", det.get("voi", 0.0))
            is_1st = det.get("is_first_in_pair", True)
            print(f"  Step {i}: Observed {step['road_id']:<16} (Cost={step['cost']:.1f})")
            print(f"    Mode: {mode} | Score={step['score']:.4f} | Partner={partner} | JointVOI={j_voi:.3f} | FirstInPair={is_1st}")
            print(f"    Reported: {step['reported'].upper()} (Truth was {'OPEN' if step['true'] else 'BLOCKED'})")
            print(f"    Post-Step Best Route: Route #{step['new_best_route']} (ExpCost={step['new_v0']:.2f} min)")

        print(f"  Chosen Route: Route #{res['final_route']} -> Realized Cost: {res['realized_cost']:.1f}m (Omniscient: {res['omniscient_cost']:.1f}m)")
        print(f"  Roads: {res['final_route_roads']}")
        print(f"  Route Truly Blocked: {res['is_final_blocked']} | Regret: {res['regret']:.1f} min")

    differs = (res_voi["steps_log"] != res_la2["steps_log"])
    print(f"\n>>> COMPARISON: Did choices differ? {'YES' if differs else 'NO'}")
    if differs:
        voi_obs = [s["road_id"] for s in res_voi["steps_log"]]
        la2_obs = [s["road_id"] for s in res_la2["steps_log"]]
        print(f"    ORACLE-VOI observed:        {voi_obs} -> Regret={res_voi['regret']:.1f} min")
        print(f"    ORACLE-Lookahead-2 observed: {la2_obs} -> Regret={res_la2['regret']:.1f} min")
    else:
        print(f"    Both strategies selected the identical observation sequence and realized regret {res_voi['regret']:.1f} min.")


def main() -> None:
    # 1. Seed 1, length_scale 2.0, k=3, source perfect
    replay_case(seed=1, length_scale=2.0, budget=3.0, budget_mode="count", source="perfect")
    replay_case(seed=1, length_scale=2.0, budget=3.0, budget_mode="cost", source="perfect")

    # 2. Seed 2, length_scale 2.0, k=3 and budget 3, source satellite
    replay_case(seed=2, length_scale=2.0, budget=3.0, budget_mode="count", source="satellite")
    replay_case(seed=2, length_scale=2.0, budget=3.0, budget_mode="cost", source="satellite")

    # 3. Seed 0, length_scale 2.0, k=5, source perfect
    replay_case(seed=0, length_scale=2.0, budget=5.0, budget_mode="count", source="perfect")
    replay_case(seed=0, length_scale=2.0, budget=5.0, budget_mode="cost", source="perfect")


if __name__ == "__main__":
    main()
