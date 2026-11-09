"""Pure functional dependency resolution and propagation engine for MemoryMesh.

Governed by docs/RESOLUTION_ENGINE.md Section 5 and docs/SPECIFICATION.md.
Contains pure graph traversal and propagation functions with no database,
network, or system clock access.

Edge Direction Convention:
If memory B depends on memory A:
    B --DEPENDS_ON--> A
Where:
- src_memory_id = B (the dependent / downstream memory)
- dst_memory_id = A (the upstream foundation / prerequisite memory)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Protocol

from memorymesh.models import EdgeType, MemoryStatus


class EdgeLike(Protocol):
    """Protocol for edge-like objects."""

    src_memory_id: int
    dst_memory_id: int
    type: EdgeType


@dataclass(frozen=True)
class PropagationItem:
    """Represents a downstream memory impacted by an upstream change."""

    memory_id: int
    path: list[int]
    upstream_id: int
    review_reason: str


def build_dependency_graph(
    edges: Sequence[EdgeLike],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build deterministic adjacency maps from DEPENDS_ON edges.

    Returns:
        tuple (dependents_map, dependencies_map):
        - dependents_map: upstream_id -> sorted list of unique dependent_ids (downstream)
        - dependencies_map: dependent_id -> sorted list of unique upstream_ids (upstream)
    """
    dependents_map: dict[int, set[int]] = {}
    dependencies_map: dict[int, set[int]] = {}

    for edge in edges:
        if edge.type != EdgeType.DEPENDS_ON:
            continue
        dependent = edge.src_memory_id
        upstream = edge.dst_memory_id

        # Skip self-referential edges
        if dependent == upstream:
            continue

        dependents_map.setdefault(upstream, set()).add(dependent)
        dependencies_map.setdefault(dependent, set()).add(upstream)

    sorted_dependents = {k: sorted(v) for k, v in dependents_map.items()}
    sorted_dependencies = {k: sorted(v) for k, v in dependencies_map.items()}
    return sorted_dependents, sorted_dependencies


def resolve_direct_dependencies(edges: Sequence[EdgeLike], memory_id: int) -> list[int]:
    """Return sorted unique IDs of memories that memory_id directly depends on.

    Edge: memory_id --DEPENDS_ON--> upstream.
    """
    _, dependencies_map = build_dependency_graph(edges)
    return dependencies_map.get(memory_id, [])


def resolve_direct_dependents(edges: Sequence[EdgeLike], memory_id: int) -> list[int]:
    """Return sorted unique IDs of memories that directly depend on memory_id.

    Edge: dependent --DEPENDS_ON--> memory_id.
    """
    dependents_map, _ = build_dependency_graph(edges)
    return dependents_map.get(memory_id, [])


def resolve_transitive_dependents(
    edges: Sequence[EdgeLike], memory_id: int
) -> list[tuple[int, list[int]]]:
    """Resolve all reachable downstream dependent memories using cycle-safe BFS.

    Args:
        edges: Set of dependency edges to traverse.
        memory_id: Upstream root memory ID.

    Returns:
        List of (dependent_id, path) tuples in deterministic discovery order,
        where path starts with memory_id and ends with dependent_id.
    """
    dependents_map, _ = build_dependency_graph(edges)

    visited: set[int] = {memory_id}
    queue: list[tuple[int, list[int]]] = [(memory_id, [memory_id])]
    result: list[tuple[int, list[int]]] = []

    while queue:
        current_id, current_path = queue.pop(0)
        direct = dependents_map.get(current_id, [])

        for dep_id in direct:
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep_path = current_path + [dep_id]
            result.append((dep_id, dep_path))
            queue.append((dep_id, dep_path))

    return result


def compute_propagation(
    edges: Sequence[EdgeLike],
    memory_statuses: Mapping[int, MemoryStatus],
    changed_ids: Sequence[int],
    cause_text: str,
    *,
    already_flagged_ids: Set[int] | None = None,
) -> list[PropagationItem]:
    """Compute downstream memories that require review following upstream changes.

    Pure deterministic propagation:
    - Traverses DEPENDS_ON edges downstream from changed_ids.
    - Avoids infinite loops if cycles exist in stored graph.
    - Identifies ACTIVE or DISPUTED memories in the downstream graph.
    - Produces explainable PropagationItem records preserving the exact dependency path.
    - Deduplicates multiple paths to the same dependent memory within this run.

    Args:
        edges: Relevant DEPENDS_ON edges in the graph.
        memory_statuses: Mapping of memory_id to current MemoryStatus.
        changed_ids: Upstream memory IDs that were changed / superseded / disputed.
        cause_text: Explanation of what happened to the upstream memory.
        already_flagged_ids: Optional set for backwards compatibility (ignored; event-level
            idempotency is governed by decision log history).

    Returns:
        List of PropagationItem to be updated and audited by the service.
    """
    dependents_map, _ = build_dependency_graph(edges)

    flagged_items: list[PropagationItem] = []

    # Traverse each changed root independently for root-cause-aware propagation
    for root_id in sorted(set(changed_ids)):
        visited: set[int] = {root_id}
        emitted_for_root: set[int] = set()

        # Queue contains (current_node, path_from_root)
        queue: list[tuple[int, list[int]]] = [(root_id, [root_id])]

        while queue:
            node, path = queue.pop(0)
            direct_dependents = dependents_map.get(node, [])

            for dep in direct_dependents:
                if dep in visited:
                    continue
                visited.add(dep)
                dep_path = path + [dep]

                dep_status = memory_statuses.get(dep)
                # Only flag memories that are active or disputed in the world
                if dep_status in (MemoryStatus.ACTIVE, MemoryStatus.DISPUTED):
                    if dep not in emitted_for_root:
                        emitted_for_root.add(dep)
                        path_str = " -> ".join(f"M{x}" for x in dep_path)
                        if len(dep_path) > 2:
                            reason = (
                                f"depends on changed upstream M{root_id} via {path_str}, "
                                f"which was {cause_text}"
                            )
                        else:
                            reason = (
                                f"depends on changed upstream M{root_id}, which was {cause_text}"
                            )
                        flagged_items.append(
                            PropagationItem(
                                memory_id=dep,
                                path=dep_path,
                                upstream_id=root_id,
                                review_reason=reason,
                            )
                        )

                queue.append((dep, dep_path))

    return flagged_items


def compute_impact_paths(
    edges: Sequence[EdgeLike],
    root_id: int,
    *,
    max_paths_per_dependent: int = 10,
    max_search_budget: int = 2000,
) -> list[tuple[int, list[int], list[list[int]], bool]]:
    """Compute reachable downstream dependent memories and simple paths from root.

    Guarantees complete reachability: all downstream dependents are discovered
    regardless of chain length, eliminating silent depth truncation.
    Phase 1 performs O(V + E) BFS with parent tracking to discover all reachable
    dependents and reconstruct exact shortest paths.

    Safeguards against exponential path explosion:
    - Traversal is cycle-safe (never revisits a node on the same path).
    - Caps total retained paths at max_paths_per_dependent per dependent
      (including the primary shortest path).
    - Strictly bounds candidate path expansions via max_search_budget. One budget unit
      represents examining one candidate outgoing edge from a path prefix. High-outdegree
      nodes cannot generate an unbounded number of queued paths in a single iteration.
    - Accurately reports whether path exploration was bounded via paths_truncated.

    Args:
        edges: Set of DEPENDS_ON edges to traverse.
        root_id: Upstream root memory ID to analyze.
        max_paths_per_dependent: Maximum total number of retained paths per dependent,
            including the primary shortest path. Must be a positive integer.
        max_search_budget: Maximum candidate path expansions during alternate-path search.
            One unit of max_search_budget represents examining one candidate outgoing edge
            from an explored path prefix. Enforces a strict resource bound on queue growth
            and exploration steps. Must be a positive integer.

    Returns:
        List of (dependent_id, primary_path, all_paths, paths_truncated) tuples
        in deterministic discovery order:
        - dependent_id: the affected dependent memory ID (root_id is excluded).
        - primary_path: the shortest path from root_id to dependent_id.
        - all_paths: sorted list of simple (acyclic) paths from root_id to dependent_id.
        - paths_truncated: True if alternate path exploration hit a budget or count limit.

    Raises:
        ValueError: If max_paths_per_dependent or max_search_budget is not a positive integer.
    """
    if (
        isinstance(max_paths_per_dependent, bool)
        or not isinstance(max_paths_per_dependent, int)
        or max_paths_per_dependent <= 0
    ):
        raise ValueError(
            f"max_paths_per_dependent must be a positive integer, got {max_paths_per_dependent!r}."
        )

    if (
        isinstance(max_search_budget, bool)
        or not isinstance(max_search_budget, int)
        or max_search_budget <= 0
    ):
        raise ValueError(
            f"max_search_budget must be a positive integer, got {max_search_budget!r}."
        )

    dependents_map, _ = build_dependency_graph(edges)

    # Phase 1: Guaranteed complete reachability & shortest path discovery via O(V + E) BFS
    discovery_order: list[int] = []
    parent: dict[int, int] = {}
    visited_nodes: set[int] = {root_id}
    bfs_queue: deque[int] = deque([root_id])

    while bfs_queue:
        curr = bfs_queue.popleft()
        for dep in dependents_map.get(curr, []):
            if dep == root_id or dep in visited_nodes:
                continue
            visited_nodes.add(dep)
            parent[dep] = curr
            discovery_order.append(dep)
            bfs_queue.append(dep)

    if not discovery_order:
        return []

    # Reconstruct shortest paths from parent map
    shortest_paths: dict[int, list[int]] = {}
    for dep in discovery_order:
        path = [dep]
        curr = dep
        while curr != root_id:
            curr = parent[curr]
            path.append(curr)
        path.reverse()
        shortest_paths[dep] = path

    # Phase 2: Controlled enumeration of alternate simple paths
    paths_by_dep: dict[int, list[list[int]]] = {
        dep: [shortest_paths[dep]] for dep in discovery_order
    }
    truncated_by_dep: dict[int, bool] = {dep: False for dep in discovery_order}

    path_queue: deque[tuple[int, list[int]]] = deque([(root_id, [root_id])])
    budget_used = 0
    budget_exhausted = False
    exhaustion_node: int | None = None

    while path_queue:
        curr, path = path_queue.popleft()
        direct_deps = dependents_map.get(curr, [])

        for dep in direct_deps:
            if budget_used >= max_search_budget:
                budget_exhausted = True
                exhaustion_node = curr
                break

            budget_used += 1

            if dep == root_id or dep in path:
                continue  # cycle-safe

            new_path = path + [dep]
            if dep in paths_by_dep:
                if new_path not in paths_by_dep[dep]:
                    if len(paths_by_dep[dep]) < max_paths_per_dependent:
                        paths_by_dep[dep].append(new_path)
                    else:
                        truncated_by_dep[dep] = True

            path_queue.append((dep, new_path))

        if budget_exhausted:
            break

    if budget_exhausted:
        pending_nodes: list[int] = []
        if exhaustion_node is not None:
            pending_nodes.append(exhaustion_node)
        pending_nodes.extend(curr_node for curr_node, _ in path_queue)
        pending_nodes = list(dict.fromkeys(pending_nodes))

        for curr_node in pending_nodes:
            if curr_node in truncated_by_dep:
                truncated_by_dep[curr_node] = True

        sub_queue: deque[int] = deque(pending_nodes)
        sub_visited = set(pending_nodes)
        while sub_queue:
            sn = sub_queue.popleft()
            for s_dep in dependents_map.get(sn, []):
                if s_dep in truncated_by_dep:
                    truncated_by_dep[s_dep] = True
                if s_dep not in sub_visited:
                    sub_visited.add(s_dep)
                    sub_queue.append(s_dep)

    result: list[tuple[int, list[int], list[list[int]], bool]] = []
    for dep_id in discovery_order:
        all_p = paths_by_dep[dep_id]
        primary = shortest_paths[dep_id]
        sorted_all = sorted(all_p, key=lambda p: (len(p), p))
        result.append((dep_id, primary, sorted_all, truncated_by_dep[dep_id]))

    return result
