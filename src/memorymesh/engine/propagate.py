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
