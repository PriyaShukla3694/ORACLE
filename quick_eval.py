"""Comprehensive evaluation and characterization of the ORACLE decision engine.

Includes:
1. Scenario generator characterization across 3 a priori settings ('reliable', 'mixed', 'harsh'):
   - Fraction of scenarios where omniscient best route is fully passable
   - Fraction of scenarios where best prior route is already optimal
   - Distribution of Step-1 VOI (mean, median, 25th, 75th, max, % with VOI > 0.01)
2. Evaluation across:
   - 100 seeds (0..99)
   - Settings: ['reliable', 'mixed', 'harsh']
   - Length scales: [0.0, 2.0]
   - Sources: ['perfect', 'satellite']
   - Cost budgets: [3, 5, 6, 10, 15]
   - Observation count budgets: [1, 2, 3, 5]
   - 6 strategies (ORACLE-VOI, Weighted-Heuristic, Highest-Uncertainty, Corridor-first, Corridor-clearance, Random)
3. Cluster bootstrap resampling by scenario seed for 95% CIs on regret, zero-regret rate, safe route rate,
   and paired differences (ORACLE minus baseline).
4. Saves raw results to results/quick_eval.csv.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from oracle_logic import (
    BaseStrategy,
    BeliefState,
    CorridorClearanceBaseFirst,
    CorridorFirstHospitalEnd,
    HighestUncertainty,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
    voi,
)
from routing import FAIL_PENALTY, best_route, enumerate_routes, get_route_edges
from scenario_data import Scenario, get_roads_list, random_scenario, sample_worlds
from simulate import run_episode


def characterize_generator(
    n_seeds: int = 100,
    n_worlds: int = 2000,
    length_scale: float = 2.0,
) -> pd.DataFrame:
    """Characterize the 3 a priori generator settings without tuning for any strategy."""
    settings = ["reliable", "mixed", "harsh"]
    records = []

    print("Characterizing scenario generator across 100 seeds for 3 a priori settings...")
    for setting in settings:
        passable_omniscient_count = 0
        prior_already_optimal_count = 0
        all_step1_vois: List[float] = []

        for seed in range(n_seeds):
            scen = random_scenario(seed=seed, length_scale=length_scale, setting=setting)
            roads = get_roads_list(scen.graph)
            T, A = enumerate_routes(scen.graph)
            R = len(T)
            if R == 0:
                continue

            # Check omniscient best route
            omniscient_cost = float("inf")
            omniscient_is_passable = False
            for r in range(R):
                r_rids = [e.road_id for e in get_route_edges(A, roads, r)]
                blocked = any(not scen._truth[rid] for rid in r_rids)
                cost = T[r] + (FAIL_PENALTY if blocked else 0.0)
                if cost < omniscient_cost:
                    omniscient_cost = cost
                    omniscient_is_passable = not blocked

            if omniscient_is_passable:
                passable_omniscient_count += 1

            # Check prior best route
            rng = np.random.default_rng(seed)
            W = sample_worlds(scen.graph, n_worlds=n_worlds, length_scale=length_scale, rng=rng)
            weights = np.ones(n_worlds) / float(n_worlds)
            belief = BeliefState(W, weights, roads, scen.graph, T, A)

            prior_best_r, _ = best_route(belief.cost, belief.weights)
            prior_rids = [e.road_id for e in get_route_edges(A, roads, prior_best_r)]
            prior_blocked = any(not scen._truth[rid] for rid in prior_rids)
            prior_realized = T[prior_best_r] + (FAIL_PENALTY if prior_blocked else 0.0)

            if abs(prior_realized - omniscient_cost) <= 1e-4:
                prior_already_optimal_count += 1

            # Step 1 VOI across all candidate roads
            for r in roads:
                v = voi(belief, r.road_id)
                all_step1_vois.append(v)

        voi_arr = np.array(all_step1_vois)
        records.append(
            {
                "Generator Setting": setting,
                "Omniscient Passable Rate (%)": f"{(passable_omniscient_count / n_seeds) * 100.0:.1f}%",
                "Prior Already Optimal Rate (%)": f"{(prior_already_optimal_count / n_seeds) * 100.0:.1f}%",
                "Mean Step-1 VOI (m)": round(float(np.mean(voi_arr)), 2),
                "Median VOI (m)": round(float(np.median(voi_arr)), 2),
                "P25 VOI (m)": round(float(np.percentile(voi_arr, 25)), 2),
                "P75 VOI (m)": round(float(np.percentile(voi_arr, 75)), 2),
                "Max VOI (m)": round(float(np.max(voi_arr)), 2),
                "Pct Roads VOI > 0.01 (%)": f"{(np.mean(voi_arr > 0.01)) * 100.0:.1f}%",
            }
        )

    return pd.DataFrame(records)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "seed",
    n_boot: int = 1500,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Perform cluster bootstrap by resampling unique cluster IDs (scenario seeds) with replacement."""
    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)
    if n_clusters <= 1:
        val = float(df[value_col].mean())
        return val, val, val

    # Pre-aggregate cluster means for high efficiency
    cluster_stats = df.groupby(cluster_col)[value_col].mean().to_numpy()

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample_indices = rng.integers(0, n_clusters, size=n_clusters)
        boot_means[b] = np.mean(cluster_stats[sample_indices])

    mean_val = float(df[value_col].mean())
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_val, lower, upper


def cluster_bootstrap_paired_diff(
    df: pd.DataFrame,
    strat_a: str,
    strat_b: str,
    value_col: str = "regret",
    cluster_col: str = "seed",
    match_cols: List[str] = ["setting", "length_scale", "budget_mode", "budget", "source"],
    n_boot: int = 1500,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute cluster bootstrap 95% CI for paired differences (Strat A - Strat B) clustered by seed."""
    piv = df.pivot_table(
        index=[cluster_col] + match_cols,
        columns="strategy",
        values=value_col,
    ).reset_index()

    if strat_a not in piv.columns or strat_b not in piv.columns:
        return 0.0, 0.0, 0.0

    piv["diff"] = piv[strat_a] - piv[strat_b]
    # Cluster mean diff by seed
    cluster_diffs = piv.groupby(cluster_col)["diff"].mean().to_numpy()
    n_clusters = len(cluster_diffs)
    if n_clusters <= 1:
        mean_d = float(np.mean(cluster_diffs))
        return mean_d, mean_d, mean_d

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample_indices = rng.integers(0, n_clusters, size=n_clusters)
        boot_means[b] = np.mean(cluster_diffs[sample_indices])

    mean_d = float(np.mean(piv["diff"]))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_d, lower, upper


def run_full_evaluation(n_seeds: int = 100) -> pd.DataFrame:
    """Run full factorial simulation across 100 seeds with cost and count budget modes."""
    settings = ["reliable", "mixed", "harsh"]
    length_scales = [0.0, 2.0]
    sources = ["perfect", "satellite"]
    cost_budgets = [3.0, 5.0, 6.0, 10.0, 15.0]
    count_budgets = [1.0, 2.0, 3.0, 5.0]
    n_worlds = 2000

    episodes: List[Dict[str, Any]] = []
    t0 = time.time()
    total_seeds = len(settings) * len(length_scales) * n_seeds
    completed_scenarios = 0

    print(f"\nRunning full evaluation across {n_seeds} seeds, {len(settings)} settings, {len(length_scales)} length scales...")

    for setting in settings:
        for length_scale in length_scales:
            for seed in range(n_seeds):
                # 1. Generate base scenario
                base_scen = random_scenario(seed=seed, length_scale=length_scale, setting=setting)
                truth_dict = dict(base_scen._truth)
                roads = get_roads_list(base_scen.graph)
                T, A = enumerate_routes(base_scen.graph)
                if len(T) == 0:
                    continue

                # Pre-sample prior worlds once per (setting, length_scale, seed)
                rng_worlds = np.random.default_rng(seed)
                W = sample_worlds(base_scen.graph, n_worlds=n_worlds, length_scale=length_scale, rng=rng_worlds)

                for source in sources:
                    acc = 0.85 if source == "satellite" else 1.0

                    # A. Cost Budgets
                    for b in cost_budgets:
                        strats: List[BaseStrategy] = [
                            OracleVOI(accuracy=acc, budget_mode="cost"),
                            WeightedHeuristic(budget_mode="cost"),
                            HighestUncertainty(),
                            CorridorFirstHospitalEnd(),
                            CorridorClearanceBaseFirst(),
                            RandomStrategy(rng=np.random.default_rng(seed)),
                        ]

                        for strat in strats:
                            sim = Scenario(base_scen.graph, dict(truth_dict), np.random.default_rng(seed))
                            res = run_episode(
                                scenario=sim,
                                strategy=strat,
                                budget=b,
                                budget_mode="cost",
                                source=source,
                                n_worlds=n_worlds,
                                length_scale=length_scale,
                                seed=seed,
                                worlds=W,
                            )
                            episodes.append(
                                {
                                    "seed": seed,
                                    "setting": setting,
                                    "length_scale": length_scale,
                                    "source": source,
                                    "budget_mode": "cost",
                                    "budget": b,
                                    "strategy": strat.name,
                                    "num_obs": res["num_observations"],
                                    "budget_spent": res["budget_spent"],
                                    "realized_cost": res["realized_cost"],
                                    "omniscient_cost": res["omniscient_cost"],
                                    "regret": res["regret"],
                                    "is_zero_regret": 1.0 if res["regret"] <= 1e-4 else 0.0,
                                    "is_safe": 1.0 if not res["is_final_route_blocked"] else 0.0,
                                }
                            )

                    # B. Observation Count Budgets (k = 1, 2, 3, 5)
                    for k in count_budgets:
                        strats = [
                            OracleVOI(accuracy=acc, budget_mode="count"),
                            WeightedHeuristic(budget_mode="count"),
                            HighestUncertainty(),
                            CorridorFirstHospitalEnd(),
                            CorridorClearanceBaseFirst(),
                            RandomStrategy(rng=np.random.default_rng(seed)),
                        ]

                        for strat in strats:
                            sim = Scenario(base_scen.graph, dict(truth_dict), np.random.default_rng(seed))
                            res = run_episode(
                                scenario=sim,
                                strategy=strat,
                                budget=k,
                                budget_mode="count",
                                source=source,
                                n_worlds=n_worlds,
                                length_scale=length_scale,
                                seed=seed,
                                worlds=W,
                            )
                            episodes.append(
                                {
                                    "seed": seed,
                                    "setting": setting,
                                    "length_scale": length_scale,
                                    "source": source,
                                    "budget_mode": "count",
                                    "budget": k,
                                    "strategy": strat.name,
                                    "num_obs": res["num_observations"],
                                    "budget_spent": res["budget_spent"],
                                    "realized_cost": res["realized_cost"],
                                    "omniscient_cost": res["omniscient_cost"],
                                    "regret": res["regret"],
                                    "is_zero_regret": 1.0 if res["regret"] <= 1e-4 else 0.0,
                                    "is_safe": 1.0 if not res["is_final_route_blocked"] else 0.0,
                                }
                            )

                completed_scenarios += 1
                if completed_scenarios % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  Completed {completed_scenarios}/{total_seeds} scenario blocks ({elapsed:.1f}s elapsed)...")

    df = pd.DataFrame(episodes)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "quick_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} total episode records to: {csv_path}")
    return df


def compute_metrics_table(df: pd.DataFrame, group_name: str = "Overall") -> pd.DataFrame:
    """Compute mean regret, zero-regret rate, and safe-route rate with cluster bootstrap 95% CIs."""
    strat_names = [
        "ORACLE-VOI",
        "Weighted-Heuristic",
        "Highest-Uncertainty",
        "Corridor-first (hospital->base)",
        "Corridor-clearance (base->hospital)",
        "Random",
    ]

    rows = []
    for s_name in strat_names:
        sub = df[df["strategy"] == s_name]
        if len(sub) == 0:
            continue

        mean_regret, lo_reg, hi_reg = cluster_bootstrap_ci(sub, "regret")
        mean_zero, lo_zero, hi_zero = cluster_bootstrap_ci(sub, "is_zero_regret")
        mean_safe, lo_safe, hi_safe = cluster_bootstrap_ci(sub, "is_safe")
        mean_cost = float(sub["realized_cost"].mean())
        mean_obs = float(sub["num_obs"].mean())

        rows.append(
            {
                "Strategy": s_name,
                "Mean Regret (95% CI)": f"{mean_regret:.2f} [{lo_reg:.2f}, {hi_reg:.2f}]",
                "Zero-Regret Rate (95% CI)": f"{mean_zero*100:.1f}% [{lo_zero*100:.1f}%, {hi_zero*100:.1f}%]",
                "Safe Route Rate (95% CI)": f"{mean_safe*100:.1f}% [{lo_safe*100:.1f}%, {hi_safe*100:.1f}%]",
                "Mean Cost": round(mean_cost, 1),
                "Mean Obs": round(mean_obs, 2),
            }
        )

    return pd.DataFrame(rows)


def compute_paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired differences (ORACLE minus baseline) with cluster bootstrap 95% CIs."""
    strat_names = [
        "Weighted-Heuristic",
        "Highest-Uncertainty",
        "Corridor-first (hospital->base)",
        "Corridor-clearance (base->hospital)",
        "Random",
    ]

    paired_rows = []
    piv = df.pivot_table(
        index=["seed", "setting", "length_scale", "budget_mode", "budget", "source"],
        columns="strategy",
        values="regret",
    ).reset_index()

    for base_name in strat_names:
        if base_name not in piv.columns:
            continue

        diffs = piv["ORACLE-VOI"] - piv[base_name]
        mean_d, lo_d, hi_d = cluster_bootstrap_paired_diff(
            df,
            strat_a="ORACLE-VOI",
            strat_b=base_name,
            value_col="regret",
            cluster_col="seed",
        )

        wins = int(np.sum(diffs < -1e-4))
        ties = int(np.sum(np.abs(diffs) <= 1e-4))
        losses = int(np.sum(diffs > 1e-4))

        paired_rows.append(
            {
                "Baseline Compared": base_name,
                "Mean Diff (min)": round(mean_d, 2),
                "95% Cluster Bootstrap CI": f"[{lo_d:.2f}, {hi_d:.2f}]",
                "ORACLE Wins": wins,
                "Ties": ties,
                "ORACLE Losses": losses,
            }
        )

    return pd.DataFrame(paired_rows)


def print_budget_observation_table(df: pd.DataFrame) -> None:
    """Print how many observations each strategy actually made per episode by budget."""
    print("\n" + "=" * 84)
    print("MEAN NUMBER OF OBSERVATIONS PER EPISODE BY BUDGET (COST VS COUNT MODE)")
    print("=" * 84)

    # Cost budgets
    df_cost = df[df["budget_mode"] == "cost"]
    obs_cost = df_cost.pivot_table(index="strategy", columns="budget", values="num_obs", aggfunc="mean").round(2)
    print("\n--- Cost Budget Mode (budget in cost units): ---")
    print(obs_cost.to_string())

    # Count budgets
    df_count = df[df["budget_mode"] == "count"]
    obs_count = df_count.pivot_table(index="strategy", columns="budget", values="num_obs", aggfunc="mean").round(2)
    print("\n--- Count Budget Mode (k observations regardless of cost): ---")
    print(obs_count.to_string())


def main() -> None:
    print("=" * 84)
    print("           ORACLE EXTENSIVE FACTORIAL EVALUATION & DIAGNOSTICS")
    print("=" * 84)

    # 1. Generator characterization
    char_df = characterize_generator(n_seeds=100, n_worlds=2000, length_scale=2.0)
    print("\n" + "=" * 84)
    print("SCENARIO GENERATOR CHARACTERIZATION (100 SEEDS)")
    print("=" * 84)
    print(char_df.to_string(index=False))

    # 2. Run full evaluation across 100 seeds
    df = run_full_evaluation(n_seeds=100)

    # 3. Budget observation counts
    print_budget_observation_table(df)

    # 4. Overall performance with cluster bootstrap CIs
    print("\n" + "=" * 84)
    print("OVERALL PERFORMANCE SUMMARY WITH 95% CLUSTER BOOTSTRAP CIs")
    print("=" * 84)
    overall_table = compute_metrics_table(df, "Overall")
    print(overall_table.to_string(index=False))

    # 5. Paired differences vs baselines
    print("\n" + "=" * 84)
    print("PAIRED DIFFERENCES: (ORACLE-VOI) MINUS (BASELINE) [95% CLUSTER BOOTSTRAP CI]")
    print("Negative difference = ORACLE achieved lower regret (better performance)")
    print("=" * 84)
    paired_table = compute_paired_table(df)
    print(paired_table.to_string(index=False))

    # 6. Subgroup breakdowns: by setting
    print("\n" + "=" * 84)
    print("SUBGROUP BREAKDOWN: BY GENERATOR SETTING")
    print("=" * 84)
    for setting in ["reliable", "mixed", "harsh"]:
        print(f"\n--- Generator Setting: {setting.upper()} ---")
        sub_df = df[df["setting"] == setting]
        p_tab = compute_paired_table(sub_df)
        print(p_tab.to_string(index=False))

    # 7. Subgroup breakdowns: by source
    print("\n" + "=" * 84)
    print("SUBGROUP BREAKDOWN: BY OBSERVATION SOURCE")
    print("=" * 84)
    for src in ["perfect", "satellite"]:
        print(f"\n--- Source: {src.upper()} ---")
        sub_df = df[df["source"] == src]
        p_tab = compute_paired_table(sub_df)
        print(p_tab.to_string(index=False))

    # 8. Subgroup breakdowns: by budget mode
    print("\n" + "=" * 84)
    print("SUBGROUP BREAKDOWN: BY BUDGET MODE (COST VS COUNT)")
    print("=" * 84)
    for b_mode in ["cost", "count"]:
        print(f"\n--- Budget Mode: {b_mode.upper()} ---")
        sub_df = df[df["budget_mode"] == b_mode]
        p_tab = compute_paired_table(sub_df)
        print(p_tab.to_string(index=False))

    print("\n" + "=" * 84)
    print("EVALUATION & DIAGNOSTICS COMPLETE")
    print("=" * 84)


if __name__ == "__main__":
    main()
