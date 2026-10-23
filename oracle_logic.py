"""ORACLE decision logic, Bayesian belief state updating, VOI calculation, and inspection strategies.

Strictly decision-theoretic: no ML models, no hidden truth access.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from routing import best_route, compute_route_costs
from scenario_data import RoadEdge


class BeliefState:
    """Represents the agent's uncertainty over possible flood states.

    Contains S simulated worlds, world probability weights, and road metadata.
    Contains NO ground truth.
    """

    def __init__(
        self,
        worlds: np.ndarray,
        weights: np.ndarray,
        edge_metadata: List[RoadEdge],
        graph: nx.Graph,
        T: np.ndarray,
        A: np.ndarray,
        observed_roads: Optional[Dict[str, str]] = None,
    ) -> None:
        self.worlds: np.ndarray = np.array(worlds, dtype=bool)  # (S, E)
        self.weights: np.ndarray = np.array(weights, dtype=float)  # (S,)
        # Ensure weights sum to 1
        sum_w = np.sum(self.weights)
        if sum_w > 0:
            self.weights /= sum_w

        self.edge_metadata: List[RoadEdge] = list(edge_metadata)
        self.road_id_to_idx: Dict[str, int] = {r.road_id: i for i, r in enumerate(self.edge_metadata)}
        self.graph: nx.Graph = graph
        self.T: np.ndarray = T  # (R,)
        self.A: np.ndarray = A  # (R, E)
        self.cost: np.ndarray = compute_route_costs(self.worlds, self.T, self.A)  # (S, R)
        self.observed_roads: Dict[str, str] = dict(observed_roads or {})

    def marginals(self) -> np.ndarray:
        """Return the weighted marginal probability P(passable) for each road, shape (E,)."""
        # (S,) @ (S, E) -> (E,)
        return self.weights @ self.worlds.astype(float)

    def effective_sample_size(self) -> float:
        """Compute effective sample size (ESS = 1 / sum(w^2))."""
        sum_sq = float(np.sum(self.weights**2))
        return 1.0 / sum_sq if sum_sq > 0 else 0.0

    def update(self, road_id: str, reported: str, accuracy: float = 1.0) -> None:
        """Bayesian update of world weights based on a reported road observation."""
        if road_id not in self.road_id_to_idx:
            raise KeyError(f"Unknown road_id '{road_id}'")

        e_idx = self.road_id_to_idx[road_id]
        if reported == "passable":
            likelihood = np.where(self.worlds[:, e_idx], accuracy, 1.0 - accuracy)
        elif reported == "blocked":
            likelihood = np.where(~self.worlds[:, e_idx], accuracy, 1.0 - accuracy)
        else:
            raise ValueError(f"Unknown report state '{reported}'. Must be 'passable' or 'blocked'.")

        updated_weights = self.weights * likelihood
        total_w = float(np.sum(updated_weights))
        if total_w <= 1e-12:
            warnings.warn(
                f"Degenerate belief update for {road_id}={reported}: total weight near zero. Preserving uniform floor.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.weights = np.ones_like(self.weights) / len(self.weights)
        else:
            self.weights = updated_weights / total_w

        self.observed_roads[road_id] = reported

        # Check effective sample size
        ess = self.effective_sample_size()
        S = len(self.weights)
        if ess < 0.10 * S:
            warnings.warn(
                f"Effective sample size {ess:.1f} dropped below 10% of total worlds ({S}).",
                RuntimeWarning,
                stacklevel=2,
            )


def exact_worlds(p_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic checker for tests: enumerates all 2^E worlds with exact product weights.

    Returns:
        W: Boolean array of shape (2^E, E).
        w: Exact joint probability array of shape (2^E,) summing to 1.
    """
    E = len(p_vector)
    # Generate all binary combinations of length E (0 = blocked, 1 = passable)
    combos = list(itertools.product([False, True], repeat=E))
    W = np.array(combos, dtype=bool)

    # Product weights under independence
    p_pass = p_vector[np.newaxis, :]  # (1, E)
    p_block = 1.0 - p_pass
    # For each world, prob = prod_e (p_e if W[s, e] else 1-p_e)
    prob_matrix = np.where(W, p_pass, p_block)
    w = np.prod(prob_matrix, axis=1)
    w /= np.sum(w)
    return W, w


def _parse_accuracy(accuracy: float | Tuple[float, float]) -> Tuple[float, float]:
    """Parse accuracy into (tpr, tnr)."""
    if isinstance(accuracy, (tuple, list)):
        return float(accuracy[0]), float(accuracy[1])
    val = float(accuracy)
    return val, val


def voi(
    belief: BeliefState,
    road_id: str,
    accuracy: float | Tuple[float, float] = 1.0,
) -> float:
    """Compute the Value of Information (VOI) for observing road_id.

    VOI = V0 - sum_{report in {passable, blocked}} min_r (unnormalized_w_report @ cost[:, r])
    where unnormalized_w_report = w * P(report | world).
    Handles correlations automatically across all candidate routes.
    Supports symmetric accuracy (float) or asymmetric (TPR, TNR) tuple.
    """
    if road_id not in belief.road_id_to_idx:
        raise KeyError(f"Unknown road_id: '{road_id}'")

    e_idx = belief.road_id_to_idx[road_id]
    _, v0 = best_route(belief.cost, belief.weights)
    tpr, tnr = _parse_accuracy(accuracy)
    s = belief.worlds[:, e_idx]

    sum_expected_min_cost = 0.0
    for report in ("passable", "blocked"):
        if report == "passable":
            likelihood = np.where(s, tpr, 1.0 - tnr)
        else:
            likelihood = np.where(s, 1.0 - tpr, tnr)

        unnormalized_w = belief.weights * likelihood
        p_report = float(np.sum(unnormalized_w))
        if p_report <= 1e-12:
            continue

        # Expected cost vector across routes for this report: (R,)
        exp_costs_for_report = unnormalized_w @ belief.cost
        min_cost_for_report = float(np.min(exp_costs_for_report))
        sum_expected_min_cost += min_cost_for_report

    voi_value = v0 - sum_expected_min_cost
    if voi_value < 1e-9:
        return 0.0
    return float(voi_value)


def joint_voi(
    belief: BeliefState,
    road_id_1: str,
    road_id_2: str,
    accuracy: float | Tuple[float, float] = 1.0,
) -> float:
    """Compute the joint Value of Information (VOI) of observing two roads simultaneously.

    Joint-VOI = V0 - sum_{(y1, y2)} min_r ( unnorm_w_{(y1, y2)} @ cost[:, r] )
    over the 4 report combinations (passable/blocked x passable/blocked).

    Guarantees:
    - Non-negativity: Joint-VOI >= 0
    - Monotonicity: Joint-VOI(e1, e2) >= max(VOI(e1), VOI(e2))
    """
    if road_id_1 not in belief.road_id_to_idx:
        raise KeyError(f"Unknown road_id_1: '{road_id_1}'")
    if road_id_2 not in belief.road_id_to_idx:
        raise KeyError(f"Unknown road_id_2: '{road_id_2}'")

    if road_id_1 == road_id_2:
        return voi(belief, road_id_1, accuracy=accuracy)

    e1_idx = belief.road_id_to_idx[road_id_1]
    e2_idx = belief.road_id_to_idx[road_id_2]
    _, v0 = best_route(belief.cost, belief.weights)
    tpr, tnr = _parse_accuracy(accuracy)

    s1 = belief.worlds[:, e1_idx]
    s2 = belief.worlds[:, e2_idx]

    lik1 = {
        "passable": np.where(s1, tpr, 1.0 - tnr),
        "blocked": np.where(s1, 1.0 - tpr, tnr),
    }
    lik2 = {
        "passable": np.where(s2, tpr, 1.0 - tnr),
        "blocked": np.where(s2, 1.0 - tpr, tnr),
    }

    sum_expected_min_cost = 0.0
    for rep1 in ("passable", "blocked"):
        for rep2 in ("passable", "blocked"):
            joint_lik = lik1[rep1] * lik2[rep2]
            unnorm_w = belief.weights * joint_lik
            p_joint = float(np.sum(unnorm_w))
            if p_joint <= 1e-12:
                continue

            exp_costs_for_joint = unnorm_w @ belief.cost
            min_cost_for_joint = float(np.min(exp_costs_for_joint))
            sum_expected_min_cost += min_cost_for_joint

    joint_voi_val = v0 - sum_expected_min_cost
    if joint_voi_val < 1e-9:
        return 0.0
    return float(joint_voi_val)


# ==============================================================================
# Inspection Strategies
# ==============================================================================


class BaseStrategy:
    """Abstract base class for road observation ranking strategies."""

    name: str = "Base"

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
        remaining_budget: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Rank candidates. Returns sorted list of (road_id, score, details_dict) descending."""
        raise NotImplementedError


class CorridorFirstHospitalEnd(BaseStrategy):
    """Corridor-first (hospital->base) static baseline.

    Ignores all beliefs. Computes the shortest BASE->HOSPITAL corridor by nominal travel_time
    on the full network and ranks roads on that corridor first in reverse path order
    (from the hospital end backward to base), then ranks remaining roads by distance to HOSPITAL.
    """

    name: str = "Corridor-first (hospital->base)"

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        hosp_pos = belief.graph.nodes["HOSPITAL"]["pos"]

        # Compute shortest path by travel_time on the network
        corridor_road_order: Dict[str, int] = {}
        if nx.has_path(belief.graph, "BASE", "HOSPITAL"):
            path = nx.shortest_path(belief.graph, "BASE", "HOSPITAL", weight="travel_time")
            # Reverse order: hospital end to base end
            rev_path_edges = []
            for i in range(len(path) - 1, 0, -1):
                u, v = path[i - 1], path[i]
                edge_data = belief.graph[u][v]
                rev_path_edges.append(edge_data["road_id"])

            for idx, rid in enumerate(rev_path_edges):
                corridor_road_order[rid] = idx

        scored = []
        for r in candidates:
            pos_u = belief.graph.nodes[r.u]["pos"]
            pos_v = belief.graph.nodes[r.v]["pos"]
            mid = ((pos_u[0] + pos_v[0]) / 2.0, (pos_u[1] + pos_v[1]) / 2.0)
            dist_to_hosp = float(np.sqrt((mid[0] - hosp_pos[0]) ** 2 + (mid[1] - hosp_pos[1]) ** 2))

            if r.road_id in corridor_road_order:
                order_idx = corridor_road_order[r.road_id]
                score = 1000.0 - float(order_idx)
                corridor_tag = f"Corridor (Hosp-end #{order_idx + 1})"
            else:
                score = 100.0 - dist_to_hosp
                corridor_tag = "Off-Corridor"

            details = {
                "corridor": corridor_tag,
                "dist_to_hospital": round(dist_to_hosp, 2),
                "cost": r.observation_cost,
            }
            scored.append((r.road_id, score, details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


class CorridorClearanceBaseFirst(BaseStrategy):
    """Corridor-clearance (base->hospital) static baseline.

    Ignores all beliefs. Computes the shortest BASE->HOSPITAL corridor by nominal travel_time
    on the full network and ranks roads on that corridor first in forward path order
    (outward from the depot/base toward hospital), then ranks remaining roads by distance from BASE.
    """

    name: str = "Corridor-clearance (base->hospital)"

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        base_pos = belief.graph.nodes["BASE"]["pos"]

        # Compute shortest path by travel_time on the network
        corridor_road_order: Dict[str, int] = {}
        if nx.has_path(belief.graph, "BASE", "HOSPITAL"):
            path = nx.shortest_path(belief.graph, "BASE", "HOSPITAL", weight="travel_time")
            # Forward order: base end outward to hospital
            fwd_path_edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = belief.graph[u][v]
                fwd_path_edges.append(edge_data["road_id"])

            for idx, rid in enumerate(fwd_path_edges):
                corridor_road_order[rid] = idx

        scored = []
        for r in candidates:
            pos_u = belief.graph.nodes[r.u]["pos"]
            pos_v = belief.graph.nodes[r.v]["pos"]
            mid = ((pos_u[0] + pos_v[0]) / 2.0, (pos_u[1] + pos_v[1]) / 2.0)
            dist_to_base = float(np.sqrt((mid[0] - base_pos[0]) ** 2 + (mid[1] - base_pos[1]) ** 2))

            if r.road_id in corridor_road_order:
                order_idx = corridor_road_order[r.road_id]
                score = 1000.0 - float(order_idx)
                corridor_tag = f"Corridor (Base-end #{order_idx + 1})"
            else:
                score = 100.0 - dist_to_base
                corridor_tag = "Off-Corridor"

            details = {
                "corridor": corridor_tag,
                "dist_from_base": round(dist_to_base, 2),
                "cost": r.observation_cost,
            }
            scored.append((r.road_id, score, details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


# Backward-compatible alias for existing callers
FixedPriority = CorridorFirstHospitalEnd




class HighestUncertainty(BaseStrategy):
    """Greedy uncertainty-sampling baseline: ranks by variance p * (1 - p)."""

    name: str = "Highest-Uncertainty"

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        marginals = belief.marginals()
        scored = []
        for r in candidates:
            e_idx = belief.road_id_to_idx[r.road_id]
            p = float(marginals[e_idx])
            variance = p * (1.0 - p)
            details = {
                "p_passable": round(p, 3),
                "variance": round(variance, 4),
                "cost": r.observation_cost,
            }
            scored.append((r.road_id, variance, details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


class RandomStrategy(BaseStrategy):
    """Random inspection baseline using seeded numpy Generator."""

    name: str = "Random"

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng(42)

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        random_scores = self.rng.uniform(0.0, 1.0, size=len(candidates))
        scored = []
        for i, r in enumerate(candidates):
            score = float(random_scores[i])
            details = {"random_score": round(score, 4), "cost": r.observation_cost}
            scored.append((r.road_id, score, details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


class WeightedHeuristic(BaseStrategy):
    """Legacy baseline combining uncertainty, route importance, and staleness urgency."""

    name: str = "Weighted-Heuristic"

    def __init__(self, budget_mode: str = "cost") -> None:
        self.budget_mode = budget_mode

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        marginals = belief.marginals()
        best_r_idx, _ = best_route(belief.cost, belief.weights)
        max_cost = max(r.observation_cost for r in belief.edge_metadata)

        scored = []
        for r in candidates:
            e_idx = belief.road_id_to_idx[r.road_id]
            p = float(marginals[e_idx])
            uncertainty = p * (1.0 - p)

            # Importance: 1.0 on best route, 0.5 on any candidate route, 0.1 otherwise
            on_best = bool(belief.A[best_r_idx, e_idx] > 0.5) if best_r_idx >= 0 else False
            on_any = bool(np.sum(belief.A[:, e_idx]) > 0.5)

            if on_best:
                importance = 1.0
            elif on_any:
                importance = 0.5
            else:
                importance = 0.1

            urgency = 0.5 + 0.5 * r.staleness
            norm_cost = 1.0 if self.budget_mode == "count" else max(1e-6, r.observation_cost / max_cost)
            score = (uncertainty * importance * urgency) / norm_cost

            details = {
                "uncertainty": round(uncertainty, 4),
                "importance": importance,
                "urgency": round(urgency, 2),
                "normalized_cost": round(norm_cost, 3),
            }
            scored.append((r.road_id, float(score), details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


class OracleVOI(BaseStrategy):
    """ORACLE policy ranking roads by Value of Information (VOI) adjusted for cost and urgency."""

    name: str = "ORACLE-VOI"

    def __init__(self, accuracy: float = 1.0, budget_mode: str = "cost") -> None:
        self.accuracy = accuracy
        self.budget_mode = budget_mode

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
        remaining_budget: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        max_cost = max(r.observation_cost for r in belief.edge_metadata)
        scored = []
        for r in candidates:
            raw_voi = voi(belief, r.road_id, accuracy=self.accuracy)
            urgency = 0.5 + 0.5 * r.staleness
            norm_cost = 1.0 if self.budget_mode == "count" else max(1e-6, r.observation_cost / max_cost)
            score = (raw_voi * urgency) / norm_cost

            details = {
                "voi": round(raw_voi, 3),
                "urgency": round(urgency, 2),
                "normalized_cost": round(norm_cost, 3),
                "cost": r.observation_cost,
                "staleness": r.staleness,
            }
            scored.append((r.road_id, float(score), details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


class OracleLookahead2(BaseStrategy):
    """ORACLE policy with 2-step lookahead evaluating joint VOI of road pairs.

    For each candidate road e (and affordable partner road f), computes the joint VOI
    of observing both over the 4 (or noisy-report) outcome combinations using the same
    sample worlds and weights.
    Score = joint VOI / (cost_e + cost_f) for the best affordable pair containing e.
    Recommends the first road of the best pair (breaking ties by higher single VOI,
    lower observation cost, then road_id).
    Falls back to single-road VOI when only one observation fits within remaining budget.
    """

    name: str = "ORACLE-Lookahead-2"

    def __init__(
        self,
        accuracy: float | Tuple[float, float] = 1.0,
        budget_mode: str = "count",
    ) -> None:
        self.accuracy = accuracy
        self.budget_mode = budget_mode

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
        remaining_budget: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        # 1. Single-road VOI for each candidate
        single_vois: Dict[str, float] = {
            r.road_id: voi(belief, r.road_id, accuracy=self.accuracy)
            for r in candidates
        }

        # 2. Check if lookahead-2 can be evaluated
        can_pair = len(candidates) >= 2
        if can_pair and remaining_budget is not None:
            if self.budget_mode == "count":
                if remaining_budget < 2.0:
                    can_pair = False
            else:
                sorted_costs = sorted(r.observation_cost for r in candidates)
                if sorted_costs[0] + sorted_costs[1] > remaining_budget:
                    can_pair = False

        if not can_pair:
            # Fall back to single-road VOI
            scored = []
            for r in candidates:
                s_voi = single_vois[r.road_id]
                c = 1.0 if self.budget_mode == "count" else r.observation_cost
                score = s_voi / max(1e-6, c)
                details = {
                    "mode": "fallback_single_voi",
                    "voi": round(s_voi, 3),
                    "cost": r.observation_cost,
                    "score": round(score, 4),
                    "is_first_in_pair": True,
                }
                scored.append((r.road_id, float(score), details))
            scored.sort(key=lambda x: (x[1], -belief.edge_metadata[belief.road_id_to_idx[x[0]]].observation_cost, x[0]), reverse=True)
            return scored

        # 3. Evaluate all affordable pairs
        pair_scores: Dict[Tuple[str, str], float] = {}
        pair_joint_vois: Dict[Tuple[str, str], float] = {}
        pair_first_roads: Dict[Tuple[str, str], str] = {}

        n_cand = len(candidates)
        for i in range(n_cand):
            r_i = candidates[i]
            cost_i = 1.0 if self.budget_mode == "count" else r_i.observation_cost
            for j in range(i + 1, n_cand):
                r_j = candidates[j]
                cost_j = 1.0 if self.budget_mode == "count" else r_j.observation_cost
                total_cost = cost_i + cost_j

                if remaining_budget is not None:
                    deduct_needed = 2.0 if self.budget_mode == "count" else (r_i.observation_cost + r_j.observation_cost)
                    if deduct_needed > remaining_budget:
                        continue

                j_voi = joint_voi(belief, r_i.road_id, r_j.road_id, accuracy=self.accuracy)
                p_score = j_voi / max(1e-6, total_cost)
                pair_key = (r_i.road_id, r_j.road_id)
                pair_scores[pair_key] = p_score
                pair_joint_vois[pair_key] = j_voi

                # Determine first road of this pair
                s_score_i = single_vois[r_i.road_id] / max(1e-6, cost_i)
                s_score_j = single_vois[r_j.road_id] / max(1e-6, cost_j)
                if s_score_i > s_score_j:
                    pair_first_roads[pair_key] = r_i.road_id
                elif s_score_j > s_score_i:
                    pair_first_roads[pair_key] = r_j.road_id
                else:
                    if cost_i < cost_j:
                        pair_first_roads[pair_key] = r_i.road_id
                    elif cost_j < cost_i:
                        pair_first_roads[pair_key] = r_j.road_id
                    else:
                        pair_first_roads[pair_key] = min(r_i.road_id, r_j.road_id)

        # 4. Score each candidate road e
        candidate_scores: List[Tuple[str, float, Dict[str, Any]]] = []
        for r in candidates:
            rid = r.road_id
            cost_r = 1.0 if self.budget_mode == "count" else r.observation_cost
            best_pair = None
            best_p_score = -1.0
            best_j_voi = 0.0
            best_partner = None

            for (r1, r2), p_score in pair_scores.items():
                if r1 == rid or r2 == rid:
                    if p_score > best_p_score:
                        best_p_score = p_score
                        best_pair = (r1, r2)
                        best_j_voi = pair_joint_vois[(r1, r2)]
                        best_partner = r2 if r1 == rid else r1

            if best_pair is not None:
                is_first = (pair_first_roads[best_pair] == rid)
                details = {
                    "mode": "lookahead_2",
                    "best_partner": best_partner,
                    "joint_voi": round(best_j_voi, 3),
                    "pair_score": round(best_p_score, 4),
                    "single_voi": round(single_vois[rid], 3),
                    "is_first_in_pair": is_first,
                    "cost": r.observation_cost,
                }
                candidate_scores.append((rid, float(best_p_score), details))
            else:
                s_score = single_vois[rid] / max(1e-6, cost_r)
                details = {
                    "mode": "fallback_single_voi",
                    "voi": round(single_vois[rid], 3),
                    "cost": r.observation_cost,
                    "score": round(s_score, 4),
                    "is_first_in_pair": True,
                }
                candidate_scores.append((rid, float(s_score), details))

        candidate_scores.sort(
            key=lambda x: (
                round(x[1], 6),
                1 if x[2].get("is_first_in_pair", False) else 0,
                round(x[2].get("single_voi", x[2].get("voi", 0.0)), 6),
                -x[2].get("cost", 1.0),
                [-ord(c) for c in x[0]],
            ),
            reverse=True,
        )
        return candidate_scores


def explain(belief: BeliefState, road_id: str, accuracy: float = 1.0) -> str:
    """Generate a plain-language explanation for why a road has value to observe.

    Constructed strictly from the belief state without any access to hidden truth.
    """
    if road_id not in belief.road_id_to_idx:
        return f"Road '{road_id}' not found in network."

    e_idx = belief.road_id_to_idx[road_id]
    road_meta = belief.edge_metadata[e_idx]
    v_voi = voi(belief, road_id, accuracy=accuracy)
    best_idx, v0 = best_route(belief.cost, belief.weights)
    marginals = belief.marginals()
    p_pass = float(marginals[e_idx])

    # Compute expected conditional costs given hypothetical reports
    cond_costs = {}
    cond_best_routes = {}
    for report in ("passable", "blocked"):
        if report == "passable":
            lik = np.where(belief.worlds[:, e_idx], accuracy, 1.0 - accuracy)
        else:
            lik = np.where(~belief.worlds[:, e_idx], accuracy, 1.0 - accuracy)

        unnorm_w = belief.weights * lik
        p_rep = float(np.sum(unnorm_w))
        if p_rep > 1e-12:
            norm_w = unnorm_w / p_rep
            r_idx, exp_c = best_route(belief.cost, norm_w)
            cond_best_routes[report] = r_idx
            cond_costs[report] = exp_c
        else:
            cond_best_routes[report] = best_idx
            cond_costs[report] = v0

    passable_c = cond_costs.get("passable", v0)
    blocked_c = cond_costs.get("blocked", v0)

    # Computed strictly from the route incidence matrix belief.A and current best route
    primary_route_idx = int(np.argmin(belief.T)) if len(belief.T) > 0 else -1
    is_on_current_best = bool(best_idx >= 0 and belief.A[best_idx, e_idx] > 0.5)
    is_on_primary = bool(primary_route_idx >= 0 and belief.A[primary_route_idx, e_idx] > 0.5)
    is_on_any_candidate = bool(np.any(belief.A[:, e_idx] > 0.5))

    if is_on_current_best:
        route_status = "Current best route (On the CURRENT BEST route)"
    elif is_on_primary:
        route_status = "Other candidate route (primary corridor) [On another candidate route]"
    elif is_on_any_candidate:
        route_status = "Other candidate route (alternate) [On another candidate route]"
    else:
        route_status = "On no route (dead end) [On no candidate route]"

    return (
        f"Recommendation Explanation for {road_meta.road_name} ({road_id}):\n"
        f"  - P(Passable): {p_pass*100:.1f}%, Staleness: {road_meta.staleness:.2f}, Observation Cost: {road_meta.observation_cost:.1f}\n"
        f"  - Route Incidence: {route_status}.\n"
        f"  - Current baseline expected journey cost: {v0:.1f} min.\n"
        f"  - If reported PASSABLE: optimal route is Route #{cond_best_routes.get('passable', 0)} (expected cost: {passable_c:.1f} min).\n"
        f"  - If reported BLOCKED:  optimal route is Route #{cond_best_routes.get('blocked', 0)} (expected cost: {blocked_c:.1f} min).\n"
        f"  - Expected Value of Information (VOI): {v_voi:.2f} min saved.\n"
        f"  - Decision impact: {'Observing this road materially alters routing decisions.' if v_voi > 0.01 else 'Observing this road does NOT alter the optimal route decision.'}"
    )
