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


def voi(belief: BeliefState, road_id: str, accuracy: float = 1.0) -> float:
    """Compute the Value of Information (VOI) for observing road_id.

    VOI = V0 - sum_{report in {passable, blocked}} min_r (unnormalized_w_report @ cost[:, r])
    where unnormalized_w_report = w * P(report | world).
    Handles correlations automatically across all candidate routes.
    """
    if road_id not in belief.road_id_to_idx:
        raise KeyError(f"Unknown road_id: '{road_id}'")

    e_idx = belief.road_id_to_idx[road_id]
    _, v0 = best_route(belief.cost, belief.weights)

    sum_expected_min_cost = 0.0
    for report in ("passable", "blocked"):
        if report == "passable":
            likelihood = np.where(belief.worlds[:, e_idx], accuracy, 1.0 - accuracy)
        else:
            likelihood = np.where(~belief.worlds[:, e_idx], accuracy, 1.0 - accuracy)

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
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Rank candidates. Returns sorted list of (road_id, score, details_dict) descending."""
        raise NotImplementedError


class FixedPriority(BaseStrategy):
    """Static corridor clearance policy: prioritizes roads along the nominal shortest BASE->HOSPITAL corridor (Route #0).

    Mimics standard emergency response procedure: clear the primary corridor outward from
    BASE before exploring secondary bypasses. Uses no belief updating or VOI.
    """

    name: str = "Fixed-Priority"

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        base_pos = belief.graph.nodes["BASE"]["pos"]
        scored = []
        for r in candidates:
            e_idx = belief.road_id_to_idx[r.road_id]
            is_on_primary = bool(belief.A[0, e_idx] > 0.5) if belief.A.shape[0] > 0 else False
            is_on_any = bool(np.sum(belief.A[:, e_idx]) > 0.5)

            # Midpoint distance to BASE
            pos_u = belief.graph.nodes[r.u]["pos"]
            pos_v = belief.graph.nodes[r.v]["pos"]
            mid = ((pos_u[0] + pos_v[0]) / 2.0, (pos_u[1] + pos_v[1]) / 2.0)
            dist_to_base = float(np.sqrt((mid[0] - base_pos[0]) ** 2 + (mid[1] - base_pos[1]) ** 2))

            # Primary corridor prioritized: score = 1000 - dist_to_base (clearing outward from base)
            # Secondary routes: score = 100 - dist_to_base
            # Off-route: score = -dist_to_base
            if is_on_primary:
                score = 1000.0 - dist_to_base
                corridor_tag = "Primary Corridor (Route #0)"
            elif is_on_any:
                score = 100.0 - dist_to_base
                corridor_tag = "Secondary/Bypass Corridor"
            else:
                score = -dist_to_base
                corridor_tag = "Off-Route / Isolated"

            details = {
                "corridor": corridor_tag,
                "dist_from_base": round(dist_to_base, 2),
                "cost": r.observation_cost,
            }
            scored.append((r.road_id, score, details))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored


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
            norm_cost = max(1e-6, r.observation_cost / max_cost)
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

    def __init__(self, accuracy: float = 1.0) -> None:
        self.accuracy = accuracy

    def rank(
        self,
        belief: BeliefState,
        candidates: List[RoadEdge],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        max_cost = max(r.observation_cost for r in belief.edge_metadata)
        scored = []
        for r in candidates:
            raw_voi = voi(belief, r.road_id, accuracy=self.accuracy)
            urgency = 0.5 + 0.5 * r.staleness
            norm_cost = max(1e-6, r.observation_cost / max_cost)
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

    is_on_primary = bool(belief.A[0, e_idx] > 0.5) if belief.A.shape[0] > 0 else False
    is_on_current_best = bool(best_idx >= 0 and belief.A[best_idx, e_idx] > 0.5)
    is_on_any = bool(np.sum(belief.A[:, e_idx]) > 0.5)

    if is_on_primary:
        if is_on_current_best:
            route_status = "Critical link on the primary corridor (Route #0), which is currently selected as optimal"
        else:
            route_status = (
                f"Critical link on the primary corridor (Route #0, travel time {belief.T[0]:.1f} min). "
                f"Under current flood uncertainty, Route #0 is deemed too risky (expected cost {belief.weights @ belief.cost[:, 0]:.1f} min), "
                f"forcing dispatchers to default to the safer detour (Route #{best_idx}, expected cost {v0:.1f} min)"
            )
    elif is_on_current_best:
        route_status = f"Link on the currently preferred route (Route #{best_idx})"
    elif is_on_any:
        route_status = "Link on an alternate or bypass corridor"
    else:
        route_status = "Isolated link; not part of any viable route from BASE to HOSPITAL"

    return (
        f"Recommendation Explanation for {road_meta.road_name} ({road_id}):\n"
        f"  - Prior P(Passable): {p_pass*100:.1f}%, Staleness: {road_meta.staleness:.2f}, Observation Cost: {road_meta.observation_cost:.1f}\n"
        f"  - Network Position: {route_status}.\n"
        f"  - Current baseline expected journey cost: {v0:.1f} min.\n"
        f"  - If reported PASSABLE: optimal route is Route #{cond_best_routes.get('passable', 0)} (expected cost: {passable_c:.1f} min).\n"
        f"  - If reported BLOCKED:  optimal route is Route #{cond_best_routes.get('blocked', 0)} (expected cost: {blocked_c:.1f} min).\n"
        f"  - Expected Value of Information (VOI): {v_voi:.2f} min saved.\n"
        f"  - Decision impact: {'Observing this road materially alters routing decisions.' if v_voi > 0.01 else 'Observing this road does NOT alter the optimal route decision.'}"
    )
