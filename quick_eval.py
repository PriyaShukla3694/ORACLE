"""Comprehensive evaluation across 100 random synthetic scenarios.

Evaluates 5 observation strategies across:
- 100 random topologies (seeds 0..99)
- Budgets: [3.0, 5.0]
- Observation sources: ['perfect', 'satellite']
- Spatial correlation length scales: [0.0 (independent), 2.0 (medium correlation)]
- Sample size S = 3000 Monte Carlo worlds
- Paired differences (ORACLE-VOI minus baseline) with 95% bootstrap confidence intervals.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from oracle_logic import (
    BaseStrategy,
    FixedPriority,
    HighestUncertainty,
    OracleVOI,
    RandomStrategy,
    WeightedHeuristic,
)
from scenario_data import Scenario, random_scenario
from simulate import run_episode


def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap mean and confidence interval for paired differences."""
    rng = np.random.default_rng(seed)
    mean_diff = float(np.mean(diffs))
    if len(diffs) <= 1:
        return mean_diff, mean_diff, mean_diff

    boot_means = np.zeros(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_diff, lower, upper


def run_evaluation() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Run full factorial evaluation across 100 scenarios and parameter configurations."""
    budgets = [3.0, 5.0]
    sources = ["perfect", "satellite"]
    length_scales = [0.0, 2.0]  # 0 = independent, 2.0 = medium correlation
    n_seeds = 100
    n_worlds = 3000

    episodes: List[Dict[str, Any]] = []
    total_configs = len(length_scales) * n_seeds * len(budgets) * len(sources)
    print(f"Starting evaluation across {total_configs} scenario-condition pairs (5 strategies each)...")
    t0 = time.time()

    counter = 0
    for length_scale in length_scales:
        scale_label = "Independent (0.0)" if length_scale <= 1e-6 else "Medium Corr (2.0)"
        for seed in range(n_seeds):
            # Generate base scenario with fixed truth
            base_scen = random_scenario(seed=seed, length_scale=length_scale)
            truth_dict = dict(base_scen._truth)

            for budget in budgets:
                for source in sources:
                    counter += 1
                    strategies: List[BaseStrategy] = [
                        FixedPriority(),
                        HighestUncertainty(),
                        RandomStrategy(rng=np.random.default_rng(seed)),
                        WeightedHeuristic(),
                        OracleVOI(accuracy=0.85 if source == "satellite" else 1.0),
                    ]

                    for strat in strategies:
                        sim = Scenario(
                            graph=base_scen.graph,
                            truth=dict(truth_dict),
                            rng=np.random.default_rng(seed),
                        )
                        res = run_episode(
                            scenario=sim,
                            strategy=strat,
                            budget=budget,
                            source=source,
                            n_worlds=n_worlds,
                            length_scale=length_scale,
                            seed=seed,
                        )

                        episodes.append(
                            {
                                "seed": seed,
                                "length_scale": length_scale,
                                "scale_label": scale_label,
                                "budget": budget,
                                "source": source,
                                "strategy": strat.name,
                                "realized_cost": res["realized_cost"],
                                "omniscient_cost": res["omniscient_cost"],
                                "regret": res["regret"],
                                "is_zero_regret": res["regret"] <= 1e-4,
                                "is_safe": not res["is_final_route_blocked"],
                                "budget_spent": res["budget_spent"],
                                "num_observations": res["num_observations"],
                            }
                        )

            if (seed + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  [{scale_label}] Finished seed {seed+1}/{n_seeds} ({elapsed:.1f}s elapsed)")

    df_episodes = pd.DataFrame(episodes)

    # Save raw episode outputs
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "quick_eval.csv"
    df_episodes.to_csv(csv_path, index=False)
    print(f"\nSaved raw episode data to: {csv_path}")

    # Compute aggregate performance summary
    summary_rows = []
    strat_names = [
        "Fixed-Priority",
        "Highest-Uncertainty",
        "Random",
        "Weighted-Heuristic",
        "ORACLE-VOI",
    ]

    for s_name in strat_names:
        sub = df_episodes[df_episodes["strategy"] == s_name]
        mean_cost = float(sub["realized_cost"].mean())
        mean_regret = float(sub["regret"].mean())
        zero_regret_rate = float(sub["is_zero_regret"].mean()) * 100.0
        safe_route_rate = float(sub["is_safe"].mean()) * 100.0

        summary_rows.append(
            {
                "Strategy": s_name,
                "Mean Realized Cost (m)": round(mean_cost, 2),
                "Mean Regret (m)": round(mean_regret, 2),
                "Zero-Regret Rate (%)": f"{zero_regret_rate:.1f}%",
                "Safe Route Rate (%)": f"{safe_route_rate:.1f}%",
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    # Compute Paired Differences vs ORACLE-VOI
    # Pivot regret on instance identifiers: (seed, length_scale, budget, source)
    pivoted_regret = df_episodes.pivot_table(
        index=["seed", "length_scale", "budget", "source"],
        columns="strategy",
        values="regret",
    ).reset_index()

    paired_rows = []
    for base_name in strat_names:
        if base_name == "ORACLE-VOI":
            continue
        # Paired difference = ORACLE - Baseline (negative means ORACLE has lower regret)
        diffs = (pivoted_regret["ORACLE-VOI"] - pivoted_regret[base_name]).to_numpy()
        mean_d, ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=2000, ci=0.95)

        oracle_wins = int(np.sum(diffs < -1e-4))
        ties = int(np.sum(np.abs(diffs) <= 1e-4))
        oracle_losses = int(np.sum(diffs > 1e-4))

        paired_rows.append(
            {
                "Baseline Compared": base_name,
                "Mean Regret Diff (min)": round(mean_d, 2),
                "95% Bootstrap CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "ORACLE Wins": oracle_wins,
                "Ties": ties,
                "ORACLE Losses": oracle_losses,
            }
        )

    df_paired = pd.DataFrame(paired_rows)

    # Identify scenarios where ORACLE lost or tied
    losses_and_ties = []
    for _, row in pivoted_regret.iterrows():
        # Check against any baseline
        diff_vs_wh = row["ORACLE-VOI"] - row["Weighted-Heuristic"]
        diff_vs_fp = row["ORACLE-VOI"] - row["Fixed-Priority"]
        diff_vs_hu = row["ORACLE-VOI"] - row["Highest-Uncertainty"]

        # Note any non-win
        if diff_vs_wh > 1e-4 or diff_vs_fp > 1e-4 or diff_vs_hu > 1e-4:
            losses_and_ties.append(
                {
                    "seed": int(row["seed"]),
                    "length_scale": row["length_scale"],
                    "budget": row["budget"],
                    "source": row["source"],
                    "oracle_regret": row["ORACLE-VOI"],
                    "weighted_heuristic_regret": row["Weighted-Heuristic"],
                    "fixed_priority_regret": row["Fixed-Priority"],
                    "highest_uncertainty_regret": row["Highest-Uncertainty"],
                }
            )

    return df_summary, df_paired, losses_and_ties


def main() -> None:
    print("=" * 82)
    print("ORACLE QUICK EVALUATION: 100 RANDOM SYNTHETIC SCENARIOS")
    print("Factorial Design: 100 seeds x 2 budgets x 2 sources x 2 length scales")
    print("=" * 82)

    df_summary, df_paired, losses_and_ties = run_evaluation()

    print("\n" + "=" * 82)
    print("AGGREGATE PERFORMANCE SUMMARY ACROSS ALL 800 RUNS PER STRATEGY")
    print("=" * 82)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 82)
    print("PAIRED DIFFERENCES: (ORACLE-VOI) MINUS (BASELINE)")
    print("Negative difference = ORACLE achieved lower regret (better performance)")
    print("=" * 82)
    print(df_paired.to_string(index=False))

    print("\n" + "=" * 82)
    print(f"ANALYSIS OF SCENARIOS WHERE ORACLE LOST OR TIED")
    print("=" * 82)
    print(f"Total instances where at least one baseline had strictly lower regret: {len(losses_and_ties)} / 800")
    if losses_and_ties:
        print("\nSample instances where a baseline had lower regret than ORACLE:")
        sample_losses = pd.DataFrame(losses_and_ties).head(8)
        print(sample_losses.to_string(index=False))

    print("\n" + "=" * 82)
    print("EVALUATION COMPLETE")
    print("=" * 82)


if __name__ == "__main__":
    main()
