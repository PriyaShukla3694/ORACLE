"""Unit and property tests for ORACLE decision engine.

Validates correctness, lack of ground-truth leakage, VOI non-negativity, exact mathematical
convergence, Bayesian updating, budget constraints, and simulation consistency.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from oracle_logic import (
    BeliefState,
    FixedPriority,
    HighestUncertainty,
    OracleLookahead2,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
    exact_worlds,
    joint_voi,
    voi,
)
from routing import FAIL_PENALTY, best_route, compute_route_costs, enumerate_routes
from scenario_data import (
    RoadEdge,
    Scenario,
    demo_scenario,
    get_roads_list,
    random_scenario,
    sample_worlds,
)
from simulate import run_episode


def test_no_leakage() -> None:
    """Requirement A: No truth leakage.

    1. BeliefState must have no attribute or reference holding ground truth.
    2. Two scenarios identical except for hidden truth must produce identical
       rankings across all strategies before any observation.
    """
    scen1 = demo_scenario(seed=42)
    # scen2 has completely inverted truth
    inverted_truth = {k: not v for k, v in scen1._truth.items()}
    scen2 = Scenario(graph=scen1.graph, truth=inverted_truth, rng=np.random.default_rng(42))

    roads = get_roads_list(scen1.graph)
    T, A = enumerate_routes(scen1.graph)

    # Verify BeliefState has no hidden truth
    W = sample_worlds(scen1.graph, n_worlds=500, length_scale=2.0, rng=np.random.default_rng(42))
    w = np.ones(500) / 500.0
    belief = BeliefState(W, w, roads, scen1.graph, T, A)

    assert not hasattr(belief, "_truth")
    assert not hasattr(belief, "truth")
    assert "truth" not in belief.__dict__

    # Both scenarios must generate identical beliefs and initial rankings
    W1 = sample_worlds(scen1.graph, n_worlds=500, length_scale=2.0, rng=np.random.default_rng(42))
    W2 = sample_worlds(scen2.graph, n_worlds=500, length_scale=2.0, rng=np.random.default_rng(42))
    belief1 = BeliefState(W1, w, roads, scen1.graph, T, A)
    belief2 = BeliefState(W2, w, roads, scen2.graph, T, A)

    strategy_factories = [
        lambda: FixedPriority(),
        lambda: HighestUncertainty(),
        lambda: RandomStrategy(rng=np.random.default_rng(99)),
        lambda: WeightedHeuristic(),
        lambda: OracleVOI(),
    ]

    for factory in strategy_factories:
        s1 = factory()
        s2 = factory()
        r1 = s1.rank(belief1, roads)
        r2 = s2.rank(belief2, roads)
        assert [x[0] for x in r1] == [x[0] for x in r2]
        assert [x[1] for x in r1] == [x[1] for x in r2]


def test_voi_non_negative_and_zero_for_unconnected_roads() -> None:
    """Requirement B: VOI >= 0 everywhere and VOI == 0 for roads on no candidate route."""
    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    W = sample_worlds(scen.graph, n_worlds=1000, length_scale=2.0, rng=np.random.default_rng(42))
    w = np.ones(1000) / 1000.0
    belief = BeliefState(W, w, roads, scen.graph, T, A)

    for r in roads:
        val = voi(belief, r.road_id)
        assert val >= 0.0, f"VOI for {r.road_id} was negative: {val}"

    # ROAD_DECOY and ROAD_HOSP_SPUR are on no BASE->HOSPITAL route
    e_decoy_idx = belief.road_id_to_idx["ROAD_DECOY"]
    e_spur_idx = belief.road_id_to_idx["ROAD_HOSP_SPUR"]
    assert np.sum(A[:, e_decoy_idx]) == 0.0
    assert np.sum(A[:, e_spur_idx]) == 0.0

    assert voi(belief, "ROAD_DECOY") == 0.0
    assert voi(belief, "ROAD_HOSP_SPUR") == 0.0


def test_exact_voi_analytical_closed_form() -> None:
    """Requirement C: On a small network with exact_worlds, voi() matches independent closed form."""
    # Construct a minimal 2-path network: BASE->HOSPITAL via Edge 0 or Edge 1.
    # Edge 2 is an isolated dead-end edge.
    g = nx.Graph()
    g.add_node("BASE", pos=(0.0, 0.0))
    g.add_node("HOSPITAL", pos=(2.0, 0.0))
    g.add_node("N1", pos=(1.0, 1.0))
    g.add_node("ISOLATED", pos=(0.0, 2.0))

    # Path 0: BASE -> N1 -> HOSPITAL (Route 0)
    # Travel time: 10 + 10 = 20, p0 = 0.60
    g.add_edge("BASE", "N1", road_id="R0", travel_time=10.0, observation_cost=2.0, staleness=0.5, p_passable=0.60)
    g.add_edge("N1", "HOSPITAL", road_id="R_FIXED", travel_time=10.0, observation_cost=1.0, staleness=0.1, p_passable=1.0)

    # Path 1: BASE -> HOSPITAL direct (Route 1)
    # Travel time: 40, p1 = 0.95
    g.add_edge("BASE", "HOSPITAL", road_id="R1", travel_time=40.0, observation_cost=2.0, staleness=0.2, p_passable=0.95)

    # Dead-end edge (not on any route)
    g.add_edge("BASE", "ISOLATED", road_id="R_DEAD", travel_time=5.0, observation_cost=1.0, staleness=0.9, p_passable=0.50)

    roads = get_roads_list(g)
    T, A = enumerate_routes(g)
    assert len(T) == 2

    # Exact enumeration of all 2^4 = 16 worlds with independent product weights
    p_vec = np.array([r.p_passable for r in roads])
    W_exact, w_exact = exact_worlds(p_vec)
    belief = BeliefState(W_exact, w_exact, roads, g, T, A)

    # Compute VOI of R0 via oracle_logic
    computed_voi = voi(belief, "R0")

    # Independent closed-form calculation:
    # Route 0 total travel time = 20. Route 0 passable iff R0 passable (prob 0.60).
    # Route 1 total travel time = 40. Route 1 passable iff R1 passable (prob 0.95).
    # ExpCost(Route 0) = 20 + 120 * (1 - 0.60) = 20 + 48 = 68.
    # ExpCost(Route 1) = 40 + 120 * (1 - 0.95) = 40 + 6 = 46.
    # Prior best route is Route 1 with V0 = 46.0.
    v0_independent = min(20.0 + FAIL_PENALTY * (1.0 - 0.60), 40.0 + FAIL_PENALTY * (1.0 - 0.95))
    assert v0_independent == pytest.approx(46.0)

    # If R0 is reported PASSABLE (prob 0.60):
    # ExpCost(Route 0 | R0 passable) = 20 + 120 * (1 - 1.0) = 20.0
    # ExpCost(Route 1 | R0 passable) = 46.0 (independent of R0)
    # Optimal choice given PASSABLE is Route 0 (cost 20.0)
    cost_given_passable = min(20.0, 46.0)

    # If R0 is reported BLOCKED (prob 0.40):
    # ExpCost(Route 0 | R0 blocked) = 20 + 120 = 140.0
    # ExpCost(Route 1 | R0 blocked) = 46.0
    # Optimal choice given BLOCKED is Route 1 (cost 46.0)
    cost_given_blocked = min(140.0, 46.0)

    expected_posterior_cost = 0.60 * cost_given_passable + 0.40 * cost_given_blocked
    expected_voi_closed_form = v0_independent - expected_posterior_cost
    # 46.0 - (0.60 * 20.0 + 0.40 * 46.0) = 46.0 - (12.0 + 18.4) = 46.0 - 30.4 = 15.6

    np.testing.assert_allclose(computed_voi, expected_voi_closed_form, atol=1e-10)
    assert abs(computed_voi - 15.6) < 1e-10


def test_sampled_worlds_convergence() -> None:
    """Requirement D: Sampled VOI converges to exact VOI under independent priors.

    Tolerance documented: With S=20,000 Monte Carlo worlds, the sampling error is
    bounded by O(1/sqrt(S)) ~ 0.5 min on a ~15 min VOI scale (tolerance atol=0.8).
    """
    g = nx.Graph()
    g.add_node("BASE", pos=(0.0, 0.0))
    g.add_node("HOSPITAL", pos=(2.0, 0.0))
    g.add_node("N1", pos=(1.0, 1.0))
    g.add_edge("BASE", "N1", road_id="R0", travel_time=10.0, observation_cost=2.0, staleness=0.5, p_passable=0.60)
    g.add_edge("N1", "HOSPITAL", road_id="R_FIXED", travel_time=10.0, observation_cost=1.0, staleness=0.1, p_passable=1.0)
    g.add_edge("BASE", "HOSPITAL", road_id="R1", travel_time=40.0, observation_cost=2.0, staleness=0.2, p_passable=0.95)

    roads = get_roads_list(g)
    T, A = enumerate_routes(g)

    # Exact VOI
    p_vec = np.array([r.p_passable for r in roads])
    W_exact, w_exact = exact_worlds(p_vec)
    exact_voi_val = voi(BeliefState(W_exact, w_exact, roads, g, T, A), "R0")

    # Sampled VOI with independent priors (length_scale=0)
    rng = np.random.default_rng(12345)
    S = 25000
    W_sampled = sample_worlds(g, n_worlds=S, length_scale=0.0, rng=rng)
    w_sampled = np.ones(S) / float(S)
    sampled_voi_val = voi(BeliefState(W_sampled, w_sampled, roads, g, T, A), "R0")

    # Must be within 0.8 min of analytical 15.6 min
    assert abs(sampled_voi_val - exact_voi_val) < 0.8, (
        f"Sampled VOI {sampled_voi_val} diverged from exact {exact_voi_val}"
    )


def test_marginals_match_priors_independent_and_correlated() -> None:
    """Requirement E: Empirical marginals of sampled worlds match priors within tolerance.

    With S=30,000 worlds, standard error is sqrt(p(1-p)/S) <= sqrt(0.25/30000) ~ 0.0029.
    Tolerance is set to 3 * SE ~ 0.02.
    """
    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    priors = np.array([r.p_passable for r in roads])

    # 1. Independent case (length_scale = 0.0)
    rng_indep = np.random.default_rng(2026)
    W_indep = sample_worlds(scen.graph, n_worlds=30000, length_scale=0.0, rng=rng_indep)
    emp_indep = np.mean(W_indep, axis=0)
    np.testing.assert_allclose(emp_indep, priors, atol=0.02)

    # 2. Correlated case (length_scale = 2.0)
    rng_corr = np.random.default_rng(2026)
    W_corr = sample_worlds(scen.graph, n_worlds=30000, length_scale=2.0, rng=rng_corr)
    emp_corr = np.mean(W_corr, axis=0)
    np.testing.assert_allclose(emp_corr, priors, atol=0.02)


def test_determinism_and_budget_constraints() -> None:
    """Requirement F: Determinism, budget never exceeded, no road observed twice."""
    for seed in (42, 101, 777):
        scen1 = demo_scenario(seed=seed, length_scale=2.0)
        scen2 = demo_scenario(seed=seed, length_scale=2.0)
        budget = 5.0

        res1 = run_episode(scen1, OracleVOI(), budget=budget, seed=seed)
        res2 = run_episode(scen2, OracleVOI(), budget=budget, seed=seed)

        # Determinism check
        assert res1["observation_log"] == res2["observation_log"]
        assert res1["realized_cost"] == res2["realized_cost"]
        assert res1["regret"] == res2["regret"]

        # Budget check
        assert res1["budget_spent"] <= budget + 1e-9
        assert res1["remaining_budget"] >= -1e-9

        # No road observed twice
        observed_rids = [x["road_id"] for x in res1["observation_log"]]
        assert len(observed_rids) == len(set(observed_rids))


def test_budget_sanity_bounds() -> None:
    """Requirement G:
    1. With budget 0, all strategies make the same initial routing decision.
    2. With budget = total road cost and perfect source, realized cost equals omniscient cost (regret = 0).
    """
    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    total_cost = sum(r.observation_cost for r in roads)

    # Budget 0 check: all strategies make identical decision without observation
    strategies = [
        FixedPriority(),
        HighestUncertainty(),
        RandomStrategy(rng=np.random.default_rng(42)),
        WeightedHeuristic(),
        OracleVOI(),
    ]

    decisions = []
    for strat in strategies:
        sim_scen = Scenario(graph=scen.graph, truth=dict(scen._truth), rng=np.random.default_rng(42))
        res = run_episode(sim_scen, strat, budget=0.0, seed=42)
        assert res["num_observations"] == 0
        decisions.append(res["final_route_idx"])

    # All agree on the prior optimal route
    assert len(set(decisions)) == 1

    # Budget = all roads, perfect source check:
    # Full observation reveals ground truth; agent must achieve 0 regret (realized == omniscient).
    sim_scen_full = Scenario(graph=scen.graph, truth=dict(scen._truth), rng=np.random.default_rng(42))
    res_full = run_episode(sim_scen_full, OracleVOI(), budget=total_cost, source="perfect", seed=42)
    assert res_full["realized_cost"] == res_full["omniscient_cost"]
    assert res_full["regret"] == 0.0


def test_fixed_priority_corridor_first_from_hospital_end() -> None:
    """Fixed-Priority must prioritize roads on the shortest corridor ordered from the hospital end."""
    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    W = sample_worlds(scen.graph, n_worlds=100, length_scale=2.0, rng=np.random.default_rng(42))
    w = np.ones(100) / 100.0
    belief = BeliefState(W, w, roads, scen.graph, T, A)

    ranked = FixedPriority().rank(belief, roads)
    top_3_rids = [x[0] for x in ranked[:3]]
    primary_rids = {roads[e].road_id for e in range(len(roads)) if A[0, e] > 0.5}
    assert set(top_3_rids) == primary_rids

    # Path from hospital end: ROAD_B (at hospital) -> ROAD_A -> ROAD_BASE_MAIN (at base)
    assert top_3_rids[0] == "ROAD_B"
    assert top_3_rids[1] == "ROAD_A"
    assert top_3_rids[2] == "ROAD_BASE_MAIN"


def test_explain_network_position_from_incidence_matrix() -> None:
    """explain() Network Position must accurately report whether a road is on CURRENT BEST, another candidate, or no candidate route."""
    from oracle_logic import explain

    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    W = sample_worlds(scen.graph, n_worlds=500, length_scale=2.0, rng=np.random.default_rng(42))
    w = np.ones(500) / 500.0
    belief = BeliefState(W, w, roads, scen.graph, T, A)

    best_r_idx, _ = best_route(belief.cost, belief.weights)
    # In demo_scenario before observation, best route is Route #1 (Detour)
    assert best_r_idx == 1

    # ROAD_DETOUR_1 is on Route #1 (CURRENT BEST route)
    exp_best = explain(belief, "ROAD_DETOUR_1")
    assert "On the CURRENT BEST route" in exp_best

    # ROAD_B is on Route #0 (another candidate route, but not the current best route)
    exp_bypass = explain(belief, "ROAD_B")
    assert "On another candidate route" in exp_bypass

    # ROAD_DECOY is on no route
    exp_noroute = explain(belief, "ROAD_DECOY")
    assert "On no candidate route" in exp_noroute


def test_uncertainty_gap_and_oracle_robustness_seeds_0_to_20() -> None:
    """Across seeds 0-20, Highest-Uncertainty top score exceeds Road B by >= 0.03, and ORACLE picks Road B."""
    for seed in range(21):
        scen = demo_scenario(seed=seed, length_scale=2.0)
        roads = get_roads_list(scen.graph)
        T, A = enumerate_routes(scen.graph)
        rng = np.random.default_rng(seed)
        W = sample_worlds(scen.graph, n_worlds=2500, length_scale=2.0, rng=rng)
        w = np.ones(2500) / 2500.0
        belief = BeliefState(W, w, roads, scen.graph, T, A)

        # Highest-Uncertainty check
        hu_ranked = HighestUncertainty().rank(belief, roads)
        top_hu_rid, top_hu_score, _ = hu_ranked[0]
        assert top_hu_rid == "ROAD_DECOY", f"Seed {seed}: Top HU road was {top_hu_rid}, expected ROAD_DECOY"

        b_hu_score = next(score for rid, score, _ in hu_ranked if rid == "ROAD_B")
        gap = top_hu_score - b_hu_score
        assert gap >= 0.03, f"Seed {seed}: Uncertainty gap {gap:.4f} is less than 0.03"

        # ORACLE check
        ov_ranked = OracleVOI().rank(belief, roads)
        assert ov_ranked[0][0] == "ROAD_B", f"Seed {seed}: ORACLE top road was {ov_ranked[0][0]}, expected ROAD_B"


def test_oracle_matches_or_beats_baselines_in_both_states() -> None:
    """In both Passable and Blocked states of Road B, ORACLE achieves <= baseline regret and cost."""
    seed = 42
    budget = 3.0
    base = demo_scenario(seed=seed)

    strategies = [FixedPriority(), HighestUncertainty(), RandomStrategy(rng=np.random.default_rng(seed)), OracleVOI()]

    for road_b_passable in (True, False):
        truth_dict = dict(base._truth)
        truth_dict["ROAD_B"] = road_b_passable
        if road_b_passable:
            truth_dict["ROAD_BASE_MAIN"] = True
            truth_dict["ROAD_A"] = True

        oracle_res = None
        baseline_regrets = []

        for strat in strategies:
            sim = Scenario(base.graph, dict(truth_dict), np.random.default_rng(seed))
            r = run_episode(sim, strat, budget=budget, seed=seed)
            if strat.name == "ORACLE-VOI":
                oracle_res = r
            else:
                baseline_regrets.append(r["regret"])

        assert oracle_res is not None
        # In both states, ORACLE achieves 0 regret on this scenario
        assert oracle_res["regret"] == 0.0
        # In both states, ORACLE regret <= all baseline regrets
        assert oracle_res["regret"] <= min(baseline_regrets)


def test_joint_voi_monotonicity_over_single_voi() -> None:
    """Property: Joint-VOI(e1, e2) >= max(VOI(e1), VOI(e2)) for both perfect and noisy sources."""
    # Test on demo scenario
    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    W = sample_worlds(scen.graph, n_worlds=1000, length_scale=2.0, rng=np.random.default_rng(42))
    w = np.ones(1000) / 1000.0
    belief = BeliefState(W, w, roads, scen.graph, T, A)

    sources = [1.0, 0.85, (0.88, 0.82)]
    pairs = [
        ("ROAD_A", "ROAD_B"),
        ("ROAD_DETOUR_1", "ROAD_DETOUR_2"),
        ("ROAD_A", "ROAD_DECOY"),
    ]

    for acc in sources:
        for r1, r2 in pairs:
            v1 = voi(belief, r1, accuracy=acc)
            v2 = voi(belief, r2, accuracy=acc)
            v_joint = joint_voi(belief, r1, r2, accuracy=acc)

            assert v_joint >= v1 - 1e-9, f"Joint VOI {v_joint} < VOI({r1}) {v1} under acc={acc}"
            assert v_joint >= v2 - 1e-9, f"Joint VOI {v_joint} < VOI({r2}) {v2} under acc={acc}"


def test_joint_voi_exact_enumeration_on_tiny_graph() -> None:
    """Requirement: Joint VOI matches an exact enumeration on a tiny graph."""
    # Minimal 2-path network
    g = nx.Graph()
    g.add_node("BASE", pos=(0.0, 0.0))
    g.add_node("HOSPITAL", pos=(2.0, 0.0))
    g.add_node("N1", pos=(1.0, 1.0))
    g.add_node("ISOLATED", pos=(0.0, 2.0))

    # Path 0: BASE -> N1 -> HOSPITAL (Route 0): travel time 20, p=0.60
    g.add_edge("BASE", "N1", road_id="R0", travel_time=10.0, observation_cost=2.0, staleness=0.5, p_passable=0.60)
    g.add_edge("N1", "HOSPITAL", road_id="R_FIXED", travel_time=10.0, observation_cost=1.0, staleness=0.1, p_passable=1.0)
    # Path 1: BASE -> HOSPITAL direct (Route 1): travel time 40, p=0.95
    g.add_edge("BASE", "HOSPITAL", road_id="R1", travel_time=40.0, observation_cost=2.0, staleness=0.2, p_passable=0.95)
    # Dead-end edge
    g.add_edge("BASE", "ISOLATED", road_id="R_DEAD", travel_time=5.0, observation_cost=1.0, staleness=0.9, p_passable=0.50)

    roads = get_roads_list(g)
    T, A = enumerate_routes(g)
    p_vec = np.array([r.p_passable for r in roads])
    W_exact, w_exact = exact_worlds(p_vec)
    belief = BeliefState(W_exact, w_exact, roads, g, T, A)

    # Theoretical exact calculations:
    # V0 = 46.0
    # Single VOI(R0) = 15.6
    # Single VOI(R1) = 4.6
    # Joint VOI(R0, R1):
    #   (P, P) prob 0.57 -> min cost 20.0
    #   (P, B) prob 0.03 -> min cost 20.0
    #   (B, P) prob 0.38 -> min cost 40.0
    #   (B, B) prob 0.02 -> min cost 140.0
    # Exp cost = 0.57*20 + 0.03*20 + 0.38*40 + 0.02*140 = 12.0 + 15.2 + 2.8 = 30.0
    # Joint VOI = 46.0 - 30.0 = 16.0
    v_joint = joint_voi(belief, "R0", "R1", accuracy=1.0)
    np.testing.assert_allclose(v_joint, 16.0, atol=1e-10)

    # Joint VOI with a dead-end road (R_DEAD) must equal single VOI of R0
    v_joint_dead = joint_voi(belief, "R0", "R_DEAD", accuracy=1.0)
    v_single_r0 = voi(belief, "R0", accuracy=1.0)
    np.testing.assert_allclose(v_joint_dead, v_single_r0, atol=1e-10)


def test_oracle_lookahead_2_determinism_and_fallback() -> None:
    """Requirement: Determinism and budget fallback of OracleLookahead2."""
    scen = demo_scenario(seed=42)
    roads = get_roads_list(scen.graph)
    T, A = enumerate_routes(scen.graph)
    W = sample_worlds(scen.graph, n_worlds=500, length_scale=2.0, rng=np.random.default_rng(42))
    w = np.ones(500) / 500.0
    belief = BeliefState(W, w, roads, scen.graph, T, A)

    strat = OracleLookahead2(accuracy=1.0, budget_mode="count")

    # Run 1 and Run 2 must be identically ordered with identical scores
    rank1 = strat.rank(belief, roads, remaining_budget=2.0)
    rank2 = strat.rank(belief, roads, remaining_budget=2.0)

    assert [x[0] for x in rank1] == [x[0] for x in rank2]
    for (_, s1, d1), (_, s2, d2) in zip(rank1, rank2):
        assert abs(s1 - s2) < 1e-9
        assert d1["joint_voi"] == d2["joint_voi"]

    # When budget only allows 1 observation, it falls back to single VOI
    rank_fb = strat.rank(belief, roads, remaining_budget=1.0)
    assert rank_fb[0][2]["mode"] == "fallback_single_voi"
    # In demo scenario, Road B has highest single VOI
    assert rank_fb[0][0] == "ROAD_B"



