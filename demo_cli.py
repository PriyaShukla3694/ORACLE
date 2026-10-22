"""ORACLE Terminal Demo CLI.

Runs a simulated flood response observation episode comparing Value of Information (VOI)
against heuristic baselines.
All data and network topologies are strictly illustrative and simulated.
"""

from __future__ import annotations

import numpy as np

from oracle_logic import (
    BeliefState,
    FixedPriority,
    HighestUncertainty,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
    explain,
)
from routing import enumerate_routes
from scenario_data import Scenario, demo_scenario, get_roads_list, sample_worlds
from simulate import benchmark_random_scenarios, compare_strategies, run_episode


def print_banner() -> None:
    print("=" * 82)
    print("         ORACLE: Value-of-Information Flood Emergency Decision Engine")
    print("                 [SIMULATION ONLY - ILLUSTRATIVE SYNTHETIC DATA]")
    print("=" * 82)
    print()


def print_road_table(roads) -> None:
    print("-" * 82)
    print("SCENARIO ROAD NETWORK (ILLUSTRATIVE, SIMULATED)")
    print("-" * 82)
    header = f"{'Road ID':<17} {'Name':<26} {'Endpoints':<14} {'Time(m)':<8} {'Cost':<6} {'Stale':<7} {'P(Pass)'}"
    print(header)
    print("-" * 82)
    for r in roads:
        endpoints = f"{r.u}->{r.v}"
        print(
            f"{r.road_id:<17} {r.road_name:<26} {endpoints:<14} "
            f"{r.travel_time:<8.1f} {r.observation_cost:<6.1f} {r.staleness:<7.2f} {r.p_passable*100:.0f}%"
        )
    print("-" * 82)
    print()


def print_initial_rankings(belief, roads, strategies) -> None:
    print("-" * 82)
    print("INITIAL ROAD OBSERVATION RECOMMENDATIONS (BEFORE ANY OBSERVATION)")
    print("-" * 82)
    for strat in strategies:
        ranked = strat.rank(belief, roads)
        print(f"\n[{strat.name}] Top Candidates:")
        for rank_idx, (rid, score, details) in enumerate(ranked[:3], 1):
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items() if k not in ("cost",))
            print(f"  {rank_idx}. {rid:<16} Score={score:>8.3f}  ({detail_str})")
    print()


def print_oracle_explanation(belief, top_oracle_road: str) -> None:
    print("-" * 82)
    print("ORACLE RECOMMENDATION EXPLANATION")
    print("-" * 82)
    print(explain(belief, top_oracle_road))
    print()


def run_scenario_case(
    scenario_factory,
    strategies,
    override_road_b: bool,
    case_title: str,
    budget: float = 3.0,
    seed: int = 42,
) -> None:
    print("-" * 82)
    print(f"DEMO CASE: {case_title.upper()}")
    print("-" * 82)

    base = scenario_factory()
    truth_dict = dict(base._truth)
    truth_dict["ROAD_B"] = override_road_b  # Explicitly set Road B truth
    # If Road B is passable, ensure other route components are open to test route unlocking
    if override_road_b:
        truth_dict["ROAD_BASE_MAIN"] = True
        truth_dict["ROAD_A"] = True

    df = compare_strategies(
        lambda: Scenario(base.graph, dict(truth_dict), np.random.default_rng(seed)),
        strategies=strategies,
        budget=budget,
        seed=seed,
    )

    print(df.to_string(index=False))
    print()


def main() -> None:
    seed = 42
    budget = 3.0

    print_banner()

    # Load demo scenario
    scenario = demo_scenario(seed=seed, length_scale=2.0)
    roads = get_roads_list(scenario.graph)
    T, A = enumerate_routes(scenario.graph)

    # Initialize initial belief for display
    W = sample_worlds(scenario.graph, n_worlds=2000, length_scale=2.0, rng=np.random.default_rng(seed))
    weights = np.ones(2000, dtype=float) / 2000.0
    belief = BeliefState(W, weights, roads, scenario.graph, T, A)

    # Strategies
    strategies = [
        FixedPriority(),
        HighestUncertainty(),
        RandomStrategy(rng=np.random.default_rng(seed)),
        WeightedHeuristic(),
        OracleVOI(),
    ]

    print_road_table(roads)
    print_initial_rankings(belief, roads, strategies)

    top_oracle_rid = OracleVOI().rank(belief, roads)[0][0]
    print_oracle_explanation(belief, top_oracle_rid)

    # Outcome 1: Road B is Passable
    run_scenario_case(
        lambda: demo_scenario(seed=seed, length_scale=2.0),
        strategies=strategies,
        override_road_b=True,
        case_title="Condition 1: Road B is Passable in Reality (Corridor Usable)",
        budget=budget,
        seed=seed,
    )

    # Outcome 2: Road B is Blocked
    run_scenario_case(
        lambda: demo_scenario(seed=seed, length_scale=2.0),
        strategies=strategies,
        override_road_b=False,
        case_title="Condition 2: Road B is Blocked in Reality (Corridor Submerged)",
        budget=budget,
        seed=seed,
    )

    # Multi-scenario Random Benchmark to separate Weighted-Heuristic and ORACLE-VOI
    print("-" * 82)
    print("MULTI-SCENARIO BENCHMARK: 15 RANDOM SYNTHETIC TOPOLOGIES (BUDGET = 4.0)")
    print("Separating ORACLE-VOI from heuristics across varied topologies and bottlenecks")
    print("-" * 82)
    bench_df = benchmark_random_scenarios(n_scenarios=15, budget=4.0, seed_start=200)
    print(bench_df.to_string(index=False))
    print("-" * 82)
    print("\n[NOTE] All numbers are synthetic simulation outputs for algorithmic demonstration.\n")


if __name__ == "__main__":
    main()
