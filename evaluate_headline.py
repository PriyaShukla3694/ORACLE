"""Headline evaluation protocol for ORACLE benchmark.

Features:
- PRIMARY: Observation-count budgets k = 1, 2, 3, 5 (strictly identical number of observations).
- SECONDARY: Cost budgets 3, 6, 10 in separate tables/files. Never pooled.
- 100 seeds x 3 settings (reliable, mixed, harsh) x 2 length scales (0, 2.0) x 2 sources (perfect, satellite).
- Strategies: ORACLE-VOI, ORACLE-Lookahead-2, Weighted-Heuristic, Corridor-first (hosp->base),
  Corridor-clearance (base->hosp), Highest-Uncertainty, Random.
- 95% Cluster bootstrap CIs (clustered by seed) for mean, median, P90, zero-regret, safe-route rates.
- Evaluates both 'All Scenarios' and 'Solvable Only' subset.
- Paired differences with cluster CIs and Wilcoxon signed-rank test on per-seed mean differences.
- Subgroup breakdown by setting, source, length scale, and k.
- Timing benchmark: Compute time per recommendation.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from oracle_logic import (
    CorridorClearanceBaseFirst,
    CorridorFirstHospitalEnd,
    HighestUncertainty,
    OracleLookahead2,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
)
from routing import FAIL_PENALTY
from scenario_data import SOURCES, Scenario, get_roads_list, random_scenario, sample_worlds
from simulate import run_episode


def evaluate_seed_chunk(args: Tuple[List[int], List[str], List[float], List[str], List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    seeds, settings, length_scales, sources, budget_configs = args
    results = []
    timing_totals: Dict[str, float] = {}
    timing_counts: Dict[str, int] = {}

    for seed in seeds:
        for setting in settings:
            for ls in length_scales:
                # 1. Instantiate base scenario once
                base_scen = random_scenario(seed=seed, setting=setting, length_scale=ls)
                graph = base_scen.graph
                truth_copy = dict(base_scen._truth)

                # Pre-sample worlds once per (seed, setting, ls)
                rng_world = np.random.default_rng(seed)
                S = 1500
                W_common = sample_worlds(graph, n_worlds=S, length_scale=ls, rng=rng_world)

                for source in sources:
                    acc = SOURCES.get(source, 1.0)
                    for b_cfg in budget_configs:
                        b_mode = b_cfg["mode"]
                        b_val = b_cfg["val"]

                        # Build strategy list for this configuration
                        strat_defs = [
                            ("ORACLE-VOI", lambda: OracleVOI(accuracy=acc, budget_mode=b_mode)),
                            ("ORACLE-Lookahead-2", lambda: OracleLookahead2(accuracy=acc, budget_mode=b_mode)),
                            ("Weighted-Heuristic", lambda: WeightedHeuristic(budget_mode=b_mode)),
                            ("Corridor-first (hospital->base)", lambda: CorridorFirstHospitalEnd()),
                            ("Corridor-clearance (base->hospital)", lambda: CorridorClearanceBaseFirst()),
                            ("Highest-Uncertainty", lambda: HighestUncertainty()),
                            ("Random", lambda: RandomStrategy(rng=np.random.default_rng(seed + 999))),
                        ]

                        for s_name, s_factory in strat_defs:
                            sim_scen = Scenario(graph=graph, truth=dict(truth_copy), rng=np.random.default_rng(seed))
                            strat = s_factory()

                            t0 = time.perf_counter()
                            res = run_episode(
                                scenario=sim_scen,
                                strategy=strat,
                                budget=b_val,
                                budget_mode=b_mode,
                                source=source,
                                n_worlds=S,
                                length_scale=ls,
                                seed=seed,
                                worlds=W_common,
                            )
                            t1 = time.perf_counter()

                            n_obs = res["num_observations"]
                            if n_obs > 0:
                                timing_totals[s_name] = timing_totals.get(s_name, 0.0) + (t1 - t0)
                                timing_counts[s_name] = timing_counts.get(s_name, 0) + n_obs

                            results.append({
                                "seed": seed,
                                "setting": setting,
                                "length_scale": ls,
                                "source": source,
                                "budget_mode": b_mode,
                                "budget": b_val,
                                "strategy": s_name,
                                "num_observations": n_obs,
                                "budget_spent": res["budget_spent"],
                                "realized_cost": res["realized_cost"],
                                "omniscient_cost": res["omniscient_cost"],
                                "regret": res["regret"],
                                "zero_regret": 1.0 if res["regret"] <= 1e-6 else 0.0,
                                "safe_route": 1.0 if not res["is_final_route_blocked"] else 0.0,
                                "omniscient_passable": 1.0 if res["omniscient_cost"] < FAIL_PENALTY else 0.0,
                            })

    return results, {k: timing_totals[k] / max(1, timing_counts[k]) for k in timing_totals}


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    metric_col: str,
    cluster_col: str = "seed",
    stat_func: str = "mean",
    n_boot: int = 1000,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """Compute 95% cluster bootstrap confidence interval clustered by seed."""
    if rng is None:
        rng = np.random.default_rng(42)

    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)

    # Compute point estimate
    vals = df[metric_col].values
    if len(vals) == 0:
        return 0.0, 0.0, 0.0

    if stat_func == "mean":
        point_est = float(np.mean(vals))
    elif stat_func == "median":
        point_est = float(np.median(vals))
    elif stat_func == "p90":
        point_est = float(np.percentile(vals, 90))
    else:
        raise ValueError(f"Unknown stat_func: {stat_func}")

    # Group values by cluster
    grouped = [group[metric_col].values for _, group in df.groupby(cluster_col)]
    boot_stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_indices = rng.integers(0, n_clusters, size=n_clusters)
        sampled_vals = np.concatenate([grouped[i] for i in sample_indices])
        if stat_func == "mean":
            boot_stats[b] = np.mean(sampled_vals)
        elif stat_func == "median":
            boot_stats[b] = np.median(sampled_vals)
        elif stat_func == "p90":
            boot_stats[b] = np.percentile(sampled_vals, 90)

    ci_lower = float(np.percentile(boot_stats, 2.5))
    ci_upper = float(np.percentile(boot_stats, 97.5))
    return point_est, ci_lower, ci_upper


def paired_cluster_bootstrap_diff(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    metric_col: str = "regret",
    cluster_col: str = "seed",
    n_boot: int = 1000,
    rng: np.random.Generator = None,
) -> Tuple[float, float, float]:
    """Compute paired (X - Y) mean difference and 95% cluster bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(42)

    diff = df_x[metric_col].values - df_y[metric_col].values
    point_diff = float(np.mean(diff))

    clusters = df_x[cluster_col].values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Group differences by cluster
    cluster_diffs = [diff[clusters == c] for c in unique_clusters]
    boot_diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_indices = rng.integers(0, n_clusters, size=n_clusters)
        sampled_diffs = np.concatenate([cluster_diffs[i] for i in sample_indices])
        boot_diffs[b] = np.mean(sampled_diffs)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return point_diff, ci_lower, ci_upper


def compute_paired_analysis(
    df: pd.DataFrame,
    target_strategy: str,
    baseline_strategies: List[str],
    cluster_col: str = "seed",
) -> List[Dict[str, Any]]:
    """Compute paired differences, cluster CIs, Wilcoxon signed-rank test, and win/tie/loss counts."""
    paired_rows = []

    # Get matching keys
    match_cols = ["seed", "setting", "length_scale", "source", "budget"]

    df_target = df[df["strategy"] == target_strategy].sort_values(match_cols).reset_index(drop=True)

    for base in baseline_strategies:
        df_base = df[df["strategy"] == base].sort_values(match_cols).reset_index(drop=True)
        if len(df_target) != len(df_base):
            continue

        # Point diff and cluster CI
        pt_diff, ci_l, ci_u = paired_cluster_bootstrap_diff(df_target, df_base, metric_col="regret", cluster_col=cluster_col)

        diffs = df_target["regret"].values - df_base["regret"].values
        wins = int(np.sum(diffs < -1e-6))
        ties = int(np.sum(np.abs(diffs) <= 1e-6))
        losses = int(np.sum(diffs > 1e-6))

        # Per-seed mean differences for Wilcoxon test
        seed_diffs = []
        for s in df_target["seed"].unique():
            mean_t = df_target[df_target["seed"] == s]["regret"].mean()
            mean_b = df_base[df_base["seed"] == s]["regret"].mean()
            seed_diffs.append(mean_t - mean_b)

        seed_diffs = np.array(seed_diffs)
        if np.all(np.abs(seed_diffs) < 1e-9):
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                w_res = stats.wilcoxon(seed_diffs, zero_method="pratt")
                w_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, p_val = 0.0, 1.0

        paired_rows.append({
            "Target": target_strategy,
            "Baseline": base,
            "Mean_Diff": round(pt_diff, 2),
            "CI_Lower": round(ci_l, 2),
            "CI_Upper": round(ci_u, 2),
            "Wins": wins,
            "Ties": ties,
            "Losses": losses,
            "Wilcoxon_W": round(w_stat, 1),
            "Wilcoxon_p": p_val,
        })

    return paired_rows


def build_summary_table(df: pd.DataFrame, subset_name: str) -> List[Dict[str, Any]]:
    summary = []
    strategies = [
        "ORACLE-VOI",
        "ORACLE-Lookahead-2",
        "Weighted-Heuristic",
        "Corridor-first (hospital->base)",
        "Corridor-clearance (base->hospital)",
        "Highest-Uncertainty",
        "Random",
    ]

    for strat in strategies:
        df_strat = df[df["strategy"] == strat]
        if len(df_strat) == 0:
            continue

        mean_r, m_l, m_u = cluster_bootstrap_ci(df_strat, "regret", stat_func="mean")
        med_r, med_l, med_u = cluster_bootstrap_ci(df_strat, "regret", stat_func="median")
        p90_r, p90_l, p90_u = cluster_bootstrap_ci(df_strat, "regret", stat_func="p90")
        z_rate, z_l, z_u = cluster_bootstrap_ci(df_strat, "zero_regret", stat_func="mean")
        s_rate, s_l, s_u = cluster_bootstrap_ci(df_strat, "safe_route", stat_func="mean")

        summary.append({
            "Subset": subset_name,
            "Strategy": strat,
            "Mean_Regret": f"{mean_r:.2f} [{m_l:.2f}, {m_u:.2f}]",
            "Median_Regret": f"{med_r:.2f} [{med_l:.2f}, {med_u:.2f}]",
            "P90_Regret": f"{p90_r:.2f} [{p90_l:.2f}, {p90_u:.2f}]",
            "Zero_Regret_Rate": f"{z_rate*100:.1f}% [{z_l*100:.1f}%, {z_u*100:.1f}%]",
            "Safe_Route_Rate": f"{s_rate*100:.1f}% [{s_l*100:.1f}%, {s_u*100:.1f}%]",
            "Mean_Cost": round(float(df_strat["realized_cost"].mean()), 2),
            "Mean_Obs": round(float(df_strat["num_observations"].mean()), 2),
        })

    return summary


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(100))
    settings = ["reliable", "mixed", "harsh"]
    length_scales = [0.0, 2.0]
    sources = ["perfect", "satellite"]

    primary_budgets = [{"mode": "count", "val": k} for k in [1.0, 2.0, 3.0, 5.0]]
    secondary_budgets = [{"mode": "cost", "val": c} for c in [3.0, 6.0, 10.0]]
    all_budgets = primary_budgets + secondary_budgets

    n_workers = min(12, mp.cpu_count() or 4)
    seed_chunks = np.array_split(seeds, n_workers)
    args_list = [
        (list(chunk), settings, length_scales, sources, all_budgets)
        for chunk in seed_chunks
    ]

    print(f"Starting parallel evaluation across {n_workers} processes (100 seeds, {len(all_budgets)} budget levels, 7 strategies)...")
    start_t = time.perf_counter()

    with mp.Pool(processes=n_workers) as pool:
        chunk_results = pool.map(evaluate_seed_chunk, args_list)

    total_time = time.perf_counter() - start_t
    print(f"Simulation completed in {total_time:.2f} seconds.")

    all_rows = []
    combined_timings: Dict[str, List[float]] = {}
    for rows, timings in chunk_results:
        all_rows.extend(rows)
        for s_name, t_val in timings.items():
            combined_timings.setdefault(s_name, []).append(t_val)

    df_full = pd.DataFrame(all_rows)

    # Separate PRIMARY and SECONDARY
    df_primary = df_full[df_full["budget_mode"] == "count"].copy()
    df_secondary = df_full[df_full["budget_mode"] == "cost"].copy()

    # Save raw CSVs
    primary_eval_path = results_dir / "headline_eval.csv"
    secondary_eval_path = results_dir / "secondary_cost_eval.csv"
    df_primary.to_csv(primary_eval_path, index=False)
    df_secondary.to_csv(secondary_eval_path, index=False)
    print(f"Saved PRIMARY evaluation dataset to {primary_eval_path} ({len(df_primary)} rows).")
    print(f"Saved SECONDARY evaluation dataset to {secondary_eval_path} ({len(df_secondary)} rows).")

    # 1. PRIMARY Summary (All Scenarios vs Solvable Only)
    summary_primary_all = build_summary_table(df_primary, "All Scenarios")
    df_primary_solvable = df_primary[df_primary["omniscient_passable"] > 0.5]
    summary_primary_solvable = build_summary_table(df_primary_solvable, "Solvable Only (Omniscient Passable)")
    df_summary_primary = pd.DataFrame(summary_primary_all + summary_primary_solvable)
    primary_summary_path = results_dir / "headline_summary.csv"
    df_summary_primary.to_csv(primary_summary_path, index=False)
    print(f"Saved PRIMARY summary table to {primary_summary_path}.")

    # 2. PRIMARY Paired Analysis (Overall + Subgroups)
    baselines = [
        "Weighted-Heuristic",
        "Corridor-first (hospital->base)",
        "Corridor-clearance (base->hospital)",
        "Highest-Uncertainty",
        "Random",
    ]

    paired_primary_all = []
    for tgt in ["ORACLE-VOI", "ORACLE-Lookahead-2"]:
        tgt_bases = baselines if tgt == "ORACLE-VOI" else baselines + ["ORACLE-VOI"]
        # Overall
        overall_rows = compute_paired_analysis(df_primary, tgt, tgt_bases)
        for r in overall_rows:
            r["Subgroup_Type"] = "Overall"
            r["Subgroup_Value"] = "All (Count Budgets)"
            paired_primary_all.append(r)

        # By Setting
        for st in settings:
            sub_df = df_primary[df_primary["setting"] == st]
            rows = compute_paired_analysis(sub_df, tgt, tgt_bases)
            for r in rows:
                r["Subgroup_Type"] = "Setting"
                r["Subgroup_Value"] = st
                paired_primary_all.append(r)

        # By Source
        for src in sources:
            sub_df = df_primary[df_primary["source"] == src]
            rows = compute_paired_analysis(sub_df, tgt, tgt_bases)
            for r in rows:
                r["Subgroup_Type"] = "Source"
                r["Subgroup_Value"] = src
                paired_primary_all.append(r)

        # By Length Scale
        for ls in length_scales:
            sub_df = df_primary[df_primary["length_scale"] == ls]
            rows = compute_paired_analysis(sub_df, tgt, tgt_bases)
            for r in rows:
                r["Subgroup_Type"] = "Length_Scale"
                r["Subgroup_Value"] = str(ls)
                paired_primary_all.append(r)

        # By Budget k
        for k in [1.0, 2.0, 3.0, 5.0]:
            sub_df = df_primary[df_primary["budget"] == k]
            rows = compute_paired_analysis(sub_df, tgt, tgt_bases)
            for r in rows:
                r["Subgroup_Type"] = "Budget_k"
                r["Subgroup_Value"] = f"k={int(k)}"
                paired_primary_all.append(r)

    df_paired_primary = pd.DataFrame(paired_primary_all)
    primary_paired_path = results_dir / "headline_paired.csv"
    df_paired_primary.to_csv(primary_paired_path, index=False)
    print(f"Saved PRIMARY paired comparison table to {primary_paired_path}.")

    # 3. SECONDARY Cost Summary & Paired
    summary_sec_all = build_summary_table(df_secondary, "All Scenarios")
    df_sec_solvable = df_secondary[df_secondary["omniscient_passable"] > 0.5]
    summary_sec_solvable = build_summary_table(df_sec_solvable, "Solvable Only (Omniscient Passable)")
    df_summary_sec = pd.DataFrame(summary_sec_all + summary_sec_solvable)
    sec_summary_path = results_dir / "secondary_cost_summary.csv"
    df_summary_sec.to_csv(sec_summary_path, index=False)
    print(f"Saved SECONDARY summary table to {sec_summary_path}.")

    paired_sec_all = []
    for tgt in ["ORACLE-VOI", "ORACLE-Lookahead-2"]:
        tgt_bases = baselines if tgt == "ORACLE-VOI" else baselines + ["ORACLE-VOI"]
        overall_rows = compute_paired_analysis(df_secondary, tgt, tgt_bases)
        for r in overall_rows:
            r["Subgroup_Type"] = "Overall"
            r["Subgroup_Value"] = "All (Cost Budgets)"
            paired_sec_all.append(r)
    df_paired_sec = pd.DataFrame(paired_sec_all)
    sec_paired_path = results_dir / "secondary_cost_paired.csv"
    df_paired_sec.to_csv(sec_paired_path, index=False)
    print(f"Saved SECONDARY paired comparison table to {sec_paired_path}.")

    # 4. Print Timing per Recommendation
    print("\n--- COMPUTE TIME PER RECOMMENDATION ---")
    for s_name, t_list in combined_timings.items():
        mean_ms = np.mean(t_list) * 1000.0
        print(f"  {s_name:<36}: {mean_ms:.2f} ms / observation")


if __name__ == "__main__":
    main()
