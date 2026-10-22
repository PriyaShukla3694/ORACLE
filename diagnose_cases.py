"""Diagnosis script replaying specific outlier cases from quick_eval step-by-step.

Investigates:
1. Seed 1, length_scale 2.0, budget 3, source perfect
2. Seed 2, length_scale 2.0, budget 3, source satellite
3. Seed 0, length_scale 2.0, budget 5, source perfect
"""

from __future__ import annotations

import numpy as np

from oracle_logic import BeliefState, OracleVOI
from routing import FAIL_PENALTY, best_route, enumerate_routes, get_route_edges
from scenario_data import SOURCES, get_roads_list, random_scenario, sample_worlds


def replay_case(seed: int, length_scale: float, budget: float, source: str) -> None:
    print("=" * 84)
    print(f"CASE DIAGNOSIS: Seed={seed}, LengthScale={length_scale}, Budget={budget}, Source={source}")
    print("=" * 84)

    # 1. Instantiate scenario
    scen = random_scenario(seed=seed, length_scale=length_scale)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    R = len(T)

    print(f"Network has {len(scen.graph.nodes)} nodes, {len(roads)} roads, {R} candidate routes.")
    print("\n--- GROUND TRUTH PASSABILITY ---")
    for r in roads:
        status = "OPEN" if scen._truth[r.road_id] else "BLOCKED"
        print(f"  {r.road_id:<16}: Prior P={r.p_passable:.2f}, Cost={r.observation_cost:.1f}, True State={status}")

    print("\n--- CANDIDATE ROUTES ---")
    for r_idx in range(R):
        r_edges = get_route_edges(A, roads, r_idx)
        r_rids = [e.road_id for e in r_edges]
        is_truly_blocked = any(not scen._truth[rid] for rid in r_rids)
        real_cost = T[r_idx] + (FAIL_PENALTY if is_truly_blocked else 0.0)
        blocked_rids = [rid for rid in r_rids if not scen._truth[rid]]
        status_str = f"BLOCKED by {blocked_rids}" if is_truly_blocked else "ALL OPEN"
        print(f"  Route #{r_idx}: Time={T[r_idx]:.1f}m | Realized={real_cost:.1f}m | Truth: {status_str} | Roads: {r_rids}")

    # Initialize Monte Carlo belief
    rng = np.random.default_rng(seed)
    W = sample_worlds(scen.graph, n_worlds=3000, length_scale=length_scale, rng=rng)
    weights = np.ones(3000, dtype=float) / 3000.0
    belief = BeliefState(W, weights, roads, scen.graph, T, A)

    r0_idx, v0 = best_route(belief.cost, belief.weights)
    print(f"\n--- INITIAL BELIEF (Before observations) ---")
    print(f"Prior optimal route: Route #{r0_idx} with Expected Cost = {v0:.2f} min")
    route_exp_costs = belief.weights @ belief.cost
    for r_idx in range(R):
        print(f"  Route #{r_idx}: ExpCost = {route_exp_costs[r_idx]:.2f} min")

    # Step-by-step observation execution
    remaining_budget = float(budget)
    step = 0
    accuracy = SOURCES.get(source, 1.0)
    strategy = OracleVOI(accuracy=accuracy)

    while remaining_budget > 0:
        candidates = [r for r in roads if r.road_id not in belief.observed_roads]
        if not candidates:
            break

        ranked = strategy.rank(belief, candidates)
        # Select best affordable
        chosen = None
        for rid, score, details in ranked:
            edge_obj = belief.edge_metadata[belief.road_id_to_idx[rid]]
            if edge_obj.observation_cost <= remaining_budget:
                chosen = (edge_obj, score, details)
                break

        if chosen is None:
            print(f"\n[Observation Step {step+1}] No remaining candidates fit budget {remaining_budget:.1f}.")
            break

        step += 1
        edge_obj, score, details = chosen
        rid = edge_obj.road_id
        obs_cost = edge_obj.observation_cost

        print(f"\n>>> STEP {step}: Remaining Budget = {remaining_budget:.1f}")
        print("  Top 3 Candidates by Oracle-VOI Score:")
        for r_rank, (c_rid, c_score, c_det) in enumerate(ranked[:3], 1):
            print(f"    {r_rank}. {c_rid:<16} Score={c_score:.4f} | VOI={c_det.get('voi', 0):.3f} | Cost={c_det.get('cost', 0):.1f}")

        # Observe
        reported = scen.observe(rid, source=source)
        remaining_budget -= obs_cost

        # Belief update
        belief.update(rid, reported, accuracy=accuracy)
        new_best_r, new_v0 = best_route(belief.cost, belief.weights)
        new_exp_costs = belief.weights @ belief.cost

        true_val = "OPEN" if scen._truth[rid] else "BLOCKED"
        print(f"  Action: Observed {rid} (Cost={obs_cost:.1f})")
        print(f"  Outcome: Reported='{reported.upper()}' (Ground Truth was '{true_val}')")
        print(f"  Post-Observation Best Route: Route #{new_best_r} with ExpCost = {new_v0:.2f} min")
        for r_idx in range(R):
            print(f"    Route #{r_idx}: ExpCost = {new_exp_costs[r_idx]:.2f} min (Marginal P(passable)={belief.marginals()[belief.road_id_to_idx[rid]]:.3f})")

    # Final Decision
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

    print("\n--- FINAL EPISODE RESULT ---")
    print(f"Chosen Route: Route #{final_r}")
    print(f"  Travel Time: {T[final_r]:.1f} min")
    print(f"  Roads along chosen route and their true states:")
    for rid in final_rids:
        observed_str = f"Observed={belief.observed_roads.get(rid, 'NO')}"
        true_str = "OPEN" if scen._truth[rid] else "BLOCKED (DISASTER!)"
        print(f"    {rid:<16}: True={true_str} | {observed_str}")

    print(f"Route Truly Blocked: {is_final_blocked}")
    print(f"Realized Cost: {final_realized:.1f} min (Omniscient Cost: {omniscient_cost:.1f} min)")
    print(f"Regret: {regret:.1f} min")
    print()


def main() -> None:
    # 1. Seed 1, length_scale 2.0, budget 3, source perfect
    replay_case(seed=1, length_scale=2.0, budget=3.0, source="perfect")

    # 2. Seed 2, length_scale 2.0, budget 3, source satellite
    replay_case(seed=2, length_scale=2.0, budget=3.0, source="satellite")

    # 3. Seed 0, length_scale 2.0, budget 5, source perfect
    replay_case(seed=0, length_scale=2.0, budget=5.0, source="perfect")


if __name__ == "__main__":
    main()
