"""Scenario definitions, road network representations, and correlated world sampler.

All flood scenario data is strictly illustrative and simulated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import cholesky
from scipy.special import ndtr

# Observation sources and their reported accuracy
SOURCES: Dict[str, float] = {
    "perfect": 1.0,
    "field": 0.95,
    "satellite": 0.85,
}


@dataclass(frozen=True)
class RoadEdge:
    """Road edge attributes."""

    road_id: str
    road_name: str
    u: str
    v: str
    travel_time: float
    observation_cost: float
    staleness: float
    p_passable: float


def get_roads_list(graph: nx.Graph) -> List[RoadEdge]:
    """Return all road edges in deterministic order sorted by road_id."""
    roads: List[RoadEdge] = []
    for u, v, data in graph.edges(data=True):
        roads.append(
            RoadEdge(
                road_id=str(data["road_id"]),
                road_name=str(data.get("road_name", data["road_id"])),
                u=str(u),
                v=str(v),
                travel_time=float(data["travel_time"]),
                observation_cost=float(data["observation_cost"]),
                staleness=float(data.get("staleness", 0.5)),
                p_passable=float(data["p_passable"]),
            )
        )
    roads.sort(key=lambda r: r.road_id)
    return roads


def sample_worlds(
    graph: nx.Graph,
    n_worlds: int,
    length_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample correlated passable/blocked worlds (S, E) using a Gaussian copula.

    True represents passable, False represents blocked.
    If length_scale <= 1e-6, roads are sampled independently.
    """
    roads = get_roads_list(graph)
    E = len(roads)
    if E == 0:
        return np.zeros((n_worlds, 0), dtype=bool)

    p_vec = np.array([r.p_passable for r in roads], dtype=float)

    if length_scale <= 1e-6:
        # Independent sampling
        uniform_samples = rng.uniform(0.0, 1.0, size=(n_worlds, E))
        return uniform_samples <= p_vec[np.newaxis, :]

    # Compute midpoints for all edges
    midpoints = np.zeros((E, 2), dtype=float)
    for i, r in enumerate(roads):
        pos_u = graph.nodes[r.u]["pos"]
        pos_v = graph.nodes[r.v]["pos"]
        midpoints[i, 0] = (pos_u[0] + pos_v[0]) / 2.0
        midpoints[i, 1] = (pos_u[1] + pos_v[1]) / 2.0

    # Distance matrix between road midpoints
    diff = midpoints[:, np.newaxis, :] - midpoints[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    # Spatial correlation covariance matrix Sigma
    sigma = np.exp(-dist / float(length_scale))
    sigma = 0.5 * (sigma + sigma.T)
    # Add regularizing jitter for numerical stability
    sigma += np.eye(E) * 1e-7

    try:
        L = cholesky(sigma, lower=True)
    except Exception:
        # Fallback to eigenvalue clipping if numerical jitter wasn't enough
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.clip(eigvals, 1e-7, None)
        sigma = (eigvecs * eigvals) @ eigvecs.T
        L = cholesky(sigma, lower=True)

    # Standard normal latent variables
    z_indep = rng.standard_normal(size=(n_worlds, E))
    z = z_indep @ L.T

    # Gaussian copula: Phi(z_i) <= p_i has marginal probability p_i
    u_vals = ndtr(z)
    return u_vals <= p_vec[np.newaxis, :]


class Scenario:
    """Simulation scenario holding network, private ground truth, and observation tracker."""

    def __init__(
        self,
        graph: nx.Graph,
        truth: Dict[str, bool],
        rng: np.random.Generator,
    ) -> None:
        self.graph = graph
        self._truth: Dict[str, bool] = dict(truth)  # Private ground truth
        self.rng = rng
        self.observed_roads: Dict[str, str] = {}  # road_id -> reported outcome

    def observe(self, road_id: str, source: str = "perfect") -> str:
        """Observe a road. Returns reported status ('passable' or 'blocked').

        Truth is never exposed directly; the returned report is noisy according
        to the observation source accuracy.
        """
        if road_id in self.observed_roads:
            raise ValueError(f"Road '{road_id}' has already been observed.")
        if road_id not in self._truth:
            raise KeyError(f"Road '{road_id}' not found in scenario truth.")
        if source not in SOURCES:
            raise ValueError(f"Unknown source '{source}'. Valid sources: {list(SOURCES.keys())}")

        accuracy = SOURCES[source]
        true_state = self._truth[road_id]

        if accuracy >= 1.0 or self.rng.random() < accuracy:
            report_state = true_state
        else:
            report_state = not true_state

        report_str = "passable" if report_state else "blocked"
        self.observed_roads[road_id] = report_str
        return report_str


def load_scenario_from_dict(
    data: Dict[str, Any],
    seed: int = 42,
    length_scale: float = 2.0,
) -> Scenario:
    """Build a Scenario from a dictionary schema."""
    rng = np.random.default_rng(seed)
    graph = nx.Graph()

    for node_id, node_attrs in data["nodes"].items():
        graph.add_node(node_id, pos=(float(node_attrs["x"]), float(node_attrs["y"])))

    for edge in data["edges"]:
        graph.add_edge(
            edge["u"],
            edge["v"],
            road_id=edge["road_id"],
            road_name=edge.get("road_name", edge["road_id"]),
            travel_time=float(edge["travel_time"]),
            observation_cost=float(edge["observation_cost"]),
            staleness=float(edge.get("staleness", 0.5)),
            p_passable=float(edge["p_passable"]),
        )

    # Sample hidden truth using the same correlated prior
    roads = get_roads_list(graph)
    truth_sample = sample_worlds(graph, n_worlds=1, length_scale=length_scale, rng=rng)[0]
    truth = {r.road_id: bool(truth_sample[i]) for i, r in enumerate(roads)}

    return Scenario(graph=graph, truth=truth, rng=rng)


def demo_scenario(seed: int = 42, length_scale: float = 2.0) -> Scenario:
    """Load the canonical illustrative demo scenario from data/demo_scenario.json."""
    json_path = Path(__file__).resolve().parent / "data" / "demo_scenario.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return load_scenario_from_dict(data, seed=seed, length_scale=length_scale)


def random_scenario(
    seed: int = 42,
    n_nodes: int = 8,
    length_scale: float = 2.0,
    setting: str = "mixed",
) -> Scenario:
    """Generate a connected synthetic geometric network with BASE and HOSPITAL far apart.

    Settings:
        - 'reliable': mostly reliable roads, Beta(5.0, 1.5) prior, mean ~0.77
        - 'mixed': default mixed conditions, Beta(2.5, 2.0) prior, mean ~0.55
        - 'harsh': severe flooding conditions, Beta(1.5, 3.5) prior, mean ~0.30
    """
    rng = np.random.default_rng(seed)
    graph = nx.Graph()

    # Place BASE at bottom-left and HOSPITAL at top-right
    graph.add_node("BASE", pos=(0.0, 0.0))
    graph.add_node("HOSPITAL", pos=(10.0, 10.0))

    # Place intermediate nodes
    node_names = ["BASE"] + [f"N_{i+1}" for i in range(n_nodes - 2)] + ["HOSPITAL"]
    for name in node_names[1:-1]:
        graph.add_node(name, pos=(float(rng.uniform(1.0, 9.0)), float(rng.uniform(1.0, 9.0))))

    # Connect nodes using distance-based Delaunay/k-nearest neighbors
    positions = {n: graph.nodes[n]["pos"] for n in node_names}
    edges_added = set()

    def add_edge_if_new(u_n: str, v_n: str) -> None:
        if u_n == v_n:
            return
        edge_key = tuple(sorted([u_n, v_n]))
        if edge_key in edges_added:
            return
        edges_added.add(edge_key)

        p1, p2 = positions[u_n], positions[v_n]
        d = float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
        travel_time = round(d * float(rng.uniform(1.0, 1.5)), 1)
        obs_cost = float(rng.integers(1, 6))
        staleness = round(float(rng.uniform(0.1, 0.9)), 2)

        # A priori passability distribution by setting
        if setting == "reliable":
            p_raw = float(rng.beta(5.0, 1.5))
            p_passable = round(max(0.10, min(0.98, p_raw)), 2)
        elif setting == "harsh":
            p_raw = float(rng.beta(1.5, 3.5))
            p_passable = round(max(0.05, min(0.90, p_raw)), 2)
        else:  # "mixed" default
            p_raw = float(rng.beta(2.5, 2.0))
            p_passable = round(max(0.05, min(0.95, p_raw)), 2)

        road_id = f"ROAD_{u_n}_{v_n}"
        graph.add_edge(
            u_n,
            v_n,
            road_id=road_id,
            road_name=f"Road {u_n}-{v_n}",
            travel_time=travel_time,
            observation_cost=obs_cost,
            staleness=staleness,
            p_passable=p_passable,
        )

    # Ensure connectedness: connect each node to its 2-3 closest neighbors
    for u_n in node_names:
        distances = []
        for v_n in node_names:
            if u_n != v_n:
                p1, p2 = positions[u_n], positions[v_n]
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                distances.append((dist, v_n))
        distances.sort(key=lambda x: x[0])
        for _, neighbor in distances[:3]:
            add_edge_if_new(u_n, neighbor)

    # Ensure BASE and HOSPITAL are connected
    if not nx.has_path(graph, "BASE", "HOSPITAL"):
        # Add intermediate bridge edges
        path_nodes = sorted(node_names[1:-1], key=lambda n: positions[n][0])
        chain = ["BASE"] + path_nodes + ["HOSPITAL"]
        for i in range(len(chain) - 1):
            add_edge_if_new(chain[i], chain[i + 1])

    roads = get_roads_list(graph)
    truth_sample = sample_worlds(graph, n_worlds=1, length_scale=length_scale, rng=rng)[0]
    truth = {r.road_id: bool(truth_sample[i]) for i, r in enumerate(roads)}

    return Scenario(graph=graph, truth=truth, rng=rng)
