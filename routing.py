"""Routing models, path enumeration, and route cost computation.

Calculates travel times, failure penalties, and expected costs across simulated worlds.
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np

from scenario_data import RoadEdge, get_roads_list

# Failure penalty added when any road on the chosen route is blocked
# Represents severe emergency delay, turnaround, and rerouting time (in minutes).
FAIL_PENALTY: float = 120.0


def enumerate_routes(
    graph: nx.Graph,
    max_routes: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate simple paths BASE -> HOSPITAL ordered by travel time.

    Returns:
        T: Route travel times vector of shape (R,).
        A: Route-edge incidence matrix of shape (R, E), where A[r, e] = 1 if road e
           is on route r, else 0.
    """
    roads = get_roads_list(graph)
    E = len(roads)
    road_to_idx = {r.road_id: i for i, r in enumerate(roads)}

    if not nx.has_path(graph, "BASE", "HOSPITAL"):
        return np.zeros((0,), dtype=float), np.zeros((0, E), dtype=float)

    path_generator = nx.shortest_simple_paths(graph, "BASE", "HOSPITAL", weight="travel_time")
    travel_times: List[float] = []
    incidence_rows: List[np.ndarray] = []

    count = 0
    for node_path in path_generator:
        row = np.zeros(E, dtype=float)
        route_time = 0.0

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = graph[u][v]
            rid = edge_data["road_id"]
            e_idx = road_to_idx[rid]
            row[e_idx] = 1.0
            route_time += float(edge_data["travel_time"])

        travel_times.append(route_time)
        incidence_rows.append(row)
        count += 1
        if count >= max_routes:
            break

    R = len(travel_times)
    if R == 0:
        return np.zeros((0,), dtype=float), np.zeros((0, E), dtype=float)

    T = np.array(travel_times, dtype=float)
    A = np.vstack(incidence_rows)
    return T, A


def compute_route_costs(
    W: np.ndarray,
    T: np.ndarray,
    A: np.ndarray,
    fail_penalty: float = FAIL_PENALTY,
) -> np.ndarray:
    """Compute route costs across simulated worlds of shape (S, R).

    cost[s, r] = T[r] + FAIL_PENALTY * 1[any road on route r is blocked in world s].
    Blocked is computed via vectorized matrix multiplication: ((~W).astype(int) @ A.T) > 0.
    """
    if W.shape[0] == 0 or T.shape[0] == 0:
        return np.zeros((W.shape[0], T.shape[0]), dtype=float)

    # (~W) is True where a road is blocked
    # Shape of blocked_counts is (S, R)
    blocked_counts = (~W).astype(int) @ A.T
    is_blocked = (blocked_counts > 0).astype(float)
    return T[np.newaxis, :] + fail_penalty * is_blocked


def best_route(cost: np.ndarray, weights: np.ndarray) -> Tuple[int, float]:
    """Find the route minimizing expected cost given world weights.

    Args:
        cost: Route costs of shape (S, R).
        weights: World probabilities/weights of shape (S,) summing to 1.

    Returns:
        Tuple of (best_route_index, expected_cost_V0).
    """
    if cost.shape[1] == 0:
        return -1, float("inf")

    # Expected cost per route: (S,) @ (S, R) -> (R,)
    expected_costs = weights @ cost
    best_idx = int(np.argmin(expected_costs))
    v0 = float(expected_costs[best_idx])
    return best_idx, v0


def get_route_edges(A: np.ndarray, roads: List[RoadEdge], route_idx: int) -> List[RoadEdge]:
    """Return the list of RoadEdge objects that make up route route_idx."""
    if route_idx < 0 or route_idx >= A.shape[0]:
        return []
    return [roads[e] for e in range(len(roads)) if A[route_idx, e] > 0.5]
