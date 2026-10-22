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
   - **Condition 2 (Road B Blocked)**: Demonstrates how ORACLE confirms the road is submerged, avoids a 139-minute disaster on the primary corridor, and safely takes the detour, matching or beating all baselines.
5. Runs a multi-scenario benchmark across 15 synthetic topologies, demonstrating that ORACLE-VOI achieves substantially lower regret than Weighted-Heuristic and standard baselines.

---

## 4. Modeling Assumptions

1. **Gaussian Copula Spatial Prior:**
   - Spatial flood dependencies between roads are simulated via a Gaussian copula with covariance $\Sigma_{ij} = \exp(-d_{ij} / \ell)$, where $d_{ij}$ is the Euclidean distance between road midpoints and $\ell$ is the correlation length scale.
   - Road marginal passability probabilities $P(\text{passable})$ exactly match input priors via the standard normal CDF transformation $\Phi(z_i) \le p_i$. When $\ell \to 0$, roads are statistically independent.

2. **Route Cost & Failure Penalty:**
   - Base cost equals the sum of edge travel times along the chosen path.
   - If any road on the chosen route is blocked in reality, an emergency failure penalty (`FAIL_PENALTY = 120.0` minutes) is added, modeling turnaround delay and forced rerouting.

3. **Value of Information (VOI):**
   - Computed as $V_0 - \sum_{m \in \{\text{passable}, \text{blocked}\}} P(m) \min_r \mathbb{E}[\text{cost}_r \mid m]$, where $V_0$ is the current best expected route cost.
   - Correctly evaluates to $0.0$ for roads that cannot alter the optimal routing decision (e.g., dead ends or roads where both outcomes leave the detour or primary route preferred).

4. **Scoring & Urgency:**
   - ORACLE score: $\text{VOI} \times \text{urgency} / \text{normalized\_cost}$, where $\text{urgency} = 0.5 + 0.5 \times \text{staleness}$ and $\text{normalized\_cost} = \text{cost} / \text{max\_cost}$.

5. **Observation Sources & Sensor Noise:**
   - **Perfect:** `accuracy = 1.0` (ideal ground truth observation).
   - **Field Inspection:** `accuracy = 0.95` (rapid field verification teams).
   - **Satellite Imagery:** Modeled with asymmetric operating characteristics reflecting remote sensing flood classification: True Positive Rate (TPR / sensitivity to passable road) = `0.88`, True Negative Rate (TNR / specificity to flood blockage) = `0.82` (mean accuracy `0.85`).
   - Observations update world weights via exact Bayesian likelihood weighting: $w \propto w_0 \cdot P(\text{report} \mid \text{state})$.

6. **ORACLE Lookahead-2 (Joint VOI):**
   - Evaluates pairs of candidate roads $(e, f)$ across all 4 joint report combinations using the same simulated worlds and Bayesian weights.
   - Computes $\text{Joint-VOI}(e, f) = V_0 - \sum_{(y_e, y_f)} P(y_e, y_f) \min_r \mathbb{E}[\text{cost}_r \mid y_e, y_f]$.
   - Proven non-negative and monotonic: $\text{Joint-VOI}(e, f) \ge \max(\text{VOI}(e), \text{VOI}(f))$.
   - Recommends the first road of the top affordable pair, falling back to myopic VOI when only a single observation fits the remaining budget.

7. **Ground Truth Isolation (No Leakage):**
   - Ground truth is private to the `Scenario` object (`_truth`) and is never accessible to `BeliefState`, routing, or ranking strategies.
   - Ground truth is only evaluated post-decision in `simulate.py` to calculate realized cost and regret.
