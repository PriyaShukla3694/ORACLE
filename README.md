# ORACLE: Value-of-Information Decision Engine for Flood Emergency Routing

> **Notice:** This project is a decision-support simulation for flood-response ideation. All road networks, flood scenarios, prior probabilities, and observation outcomes are **strictly synthetic, simulated, and illustrative**. No real-world validation or machine learning models are claimed or used.

---

## 1. Overview

During extreme flood events, emergency coordinators need to decide which uncertain roads to observe (e.g., via UAVs, satellite tasking, or field teams) under a constrained observation budget.

Traditional approaches prioritize roads with the highest uncertainty ($p(1-p)$) or roads closest to critical infrastructure. **ORACLE** computes the **Value of Information (VOI)**: the expected reduction in journey time / failure penalty achieved by resolving uncertainty on a given road before committing to an emergency route. Roads with high uncertainty that do not affect the routing decision receive zero VOI.

---

## 2. Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Virtual environment (`.venv`)

### Installation
```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `networkx`, `scipy`, `pytest`.

---

## 3. How to Run

### Run Unit and Property Tests
```bash
pytest -v
```

### Run Terminal Demo
```bash
python demo_cli.py
```

The terminal demo:
1. Prints the illustrative flood road network and prior probabilities.
2. Compares initial recommendations across 5 policies (Fixed-Priority Primary Corridor Clearance, Highest-Uncertainty, Random, Weighted-Heuristic, and ORACLE-VOI).
3. Prints a plain-language explanation of why ORACLE recommends its top road.
4. Executes full episodes under both ground truth states:
   - **Condition 1 (Road B Passable)**: Demonstrates how ORACLE safely unlocks the 19-minute primary route, saving 26 minutes over conservative detour-bound baselines.
   - **Condition 2 (Road B Blocked)**: Demonstrates how ORACLE confirms the road is blocked by flood waters, avoids a 139-minute disaster on the primary corridor, and safely takes the detour, matching or beating all baselines.
5. Runs a multi-scenario benchmark across 15 synthetic topologies, demonstrating that ORACLE-VOI achieves substantially lower regret than Weighted-Heuristic and standard baselines.

---

## 4. Modeling Assumptions

1. **Environmental Settings:**
   - **Reliable:** High road reliability; ~84% of scenarios have at least one passable route.
   - **Mixed:** Moderate disruption; ~56% of scenarios have at least one passable route.
   - **Harsh:** Severe disruption where only ~17% of scenarios have any passable route from BASE to HOSPITAL (omniscient passable rate ~17%).

2. **Gaussian Copula Spatial Prior:**
   - Spatial flood dependencies between roads are simulated via a Gaussian copula with covariance $\Sigma_{ij} = \exp(-d_{ij} / \ell)$, where $d_{ij}$ is the Euclidean distance between road midpoints and $\ell$ is the correlation length scale.
   - Road marginal passability probabilities $P(\text{passable})$ exactly match input priors via the standard normal CDF transformation $\Phi(z_i) \le p_i$. When $\ell \to 0$, roads are statistically independent.

3. **Route Cost & Failure Penalty:**
   - Base cost equals the sum of edge travel times along the chosen path.
   - If any road on the chosen route is blocked in reality, an emergency failure penalty (`FAIL_PENALTY = 120.0` minutes) is added, modeling turnaround delay and forced rerouting.

4. **Value of Information (VOI):**
   - Computed as $V_0 - \sum_{m \in \{\text{passable}, \text{blocked}\}} P(m) \min_r \mathbb{E}[\text{cost}_r \mid m]$, where $V_0$ is the current best expected route cost.
   - Correctly evaluates to $0.0$ for roads that cannot alter the optimal routing decision (e.g., dead ends or roads where both outcomes leave the detour or primary route preferred).

5. **Scoring & Urgency:**
   - ORACLE score: $\text{VOI} \times \text{urgency} / \text{normalized\_cost}$, where $\text{urgency} = 0.5 + 0.5 \times \text{staleness}$ and $\text{normalized\_cost} = \text{cost} / \text{max\_cost}$.

6. **Observation Sources & Sensor Noise:**
   - **Perfect:** `accuracy = 1.0` (ideal ground truth observation).
   - **Field Inspection:** `accuracy = 0.95` (rapid field verification teams).
   - **Satellite Imagery:** Modeled with asymmetric operating characteristics reflecting remote sensing flood classification: True Positive Rate (TPR / sensitivity to passable road) = `0.88`, True Negative Rate (TNR / specificity to flood blockage) = `0.82` (mean accuracy `0.85`).
   - Observations update world weights via exact Bayesian likelihood weighting: $w \propto w_0 \cdot P(\text{report} \mid \text{state})$.

7. **ORACLE Lookahead-2 (Joint VOI):**
   - Evaluates pairs of candidate roads $(e, f)$ across all 4 joint report combinations using the same simulated worlds and Bayesian weights.
   - Computes $\text{Joint-VOI}(e, f) = V_0 - \sum_{(y_e, y_f)} P(y_e, y_f) \min_r \mathbb{E}[\text{cost}_r \mid y_e, y_f]$.
   - Proven non-negative and monotonic: $\text{Joint-VOI}(e, f) \ge \max(\text{VOI}(e), \text{VOI}(f))$.
   - Recommends the first road of the top affordable pair, falling back to myopic VOI when only a single observation fits the remaining budget.

8. **Ground Truth Isolation (No Leakage):**
   - Ground truth is private to the `Scenario` object (`_truth`) and is never accessible to `BeliefState`, routing, or ranking strategies.
   - Ground truth is only evaluated post-decision in `simulate.py` to calculate realized cost and regret.

---

## 5. Benchmark Results & Dashboard Specification (Tab 3 & Tab 4)

### Strategy Designations
- **Default Recommended Strategy:** `ORACLE-VOI` (single-step myopic Value of Information).
- **ORACLE-Lookahead-2:** Labeled as *"Exploratory: advantage over single-step VOI not established in mean regret"*. While Lookahead-2 provides slight marginal reductions in catastrophic failure (21.6% vs 22.0%), its mean regret improvement over single-step VOI (-0.36 min) is not established beyond exploratory bounds.

### Tab 3: Evaluation & Benchmark Results Specification
- **Headline Primary Metrics:**
  - **Catastrophic-Failure Rate** (chosen route blocked in reality, incurring the $+120$ min turnaround failure penalty).
  - **Safe-Route Rate** (chosen route completely passable in reality).
  - Evaluated on the **Solvable-Only** subset under **observation-count budget mode** ($k = 1, 2, 3, 5$), with 95% cluster bootstrap confidence intervals (clustered by scenario seed).
  - Paired ORACLE-minus-baseline differences with cluster CIs and Holm-adjusted $p$-values loaded from `results/headline_paired.csv`.
  - Mean regret and median regret displayed as secondary metrics.
- **Reference & Diagnostic Tables:**
  - **All-Scenarios Reference Table:** Comprehensive performance across all generated topologies including unnavigable networks.
  - **Corridor-First vs ORACLE Zero-Regret Cross-Tabulation:** Explains why static Corridor-first achieves slightly higher exact-zero-regret episodes (39.7% vs 37.7% / 33.3%) by stubbornly holding the prior corridor (which is optimal in 22.0% of solvable cases), but suffers substantially worse catastrophic failures (30.5% vs 21.6% for Lookahead-2 and 22.0% for VOI).
  - **Subgroup Breakdowns:** All environmental setting, sensor source, correlation length scale, and budget-level breakdowns are explicitly labeled as *"Exploratory Subgroups"*, with family-wise error rates controlled via Holm step-down adjustment.

### Tab 4: Limitations & Decision Boundary Discussion
1. **Synthetic Data Boundaries:** All results derive strictly from the synthetic copula scenario generator; no real-world empirical flood validation has been conducted to date.
2. **Incomplete Information & Residual Failures:** Under constrained budgets, unobserved links can still cause catastrophic failure (~22.0% catastrophic failure rate in solvable scenarios at an average of ~2.75 observations).
3. **Severe Disruption Limits:** In the **Harsh** setting (where only ~17% of scenarios possess any passable route), heuristic importance weighting on surviving paths achieves near parity with myopic VOI.
4. **Scope:** Decision-support simulation only; no claims of real-time emergency dispatch readiness are made.
