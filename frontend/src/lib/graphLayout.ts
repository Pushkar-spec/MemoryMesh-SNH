import { MarkerType, type Node, type Edge } from '@xyflow/react';
import type { GraphNode, GraphEdge } from '../api/client';

/**
 * Transforms backend API graph nodes into React Flow nodes with deterministic DAG layout.
 * Entity nodes are positioned on the left column (x=50), and their associated memory nodes
 * are arranged in columns to the right (x=360 for active, x=680 for historical/rejected).
 * Any orphan memory nodes without matching entities are positioned below.
 */
export function layoutGraphNodes(
  apiNodes: GraphNode[] = [],
  apiEdges: GraphEdge[] = []
): Node[] {
  const entityNodes = (apiNodes ?? []).filter((n) => n.kind === 'entity');
  const memoryNodes = (apiNodes ?? []).filter((n) => n.kind === 'memory');

  const memoryToEntityMap = new Map<string, string>();
  for (const edge of apiEdges ?? []) {
    if (edge.type === 'about') {
      memoryToEntityMap.set(edge.source, edge.target);
    }
  }

  const entityMemoriesMap = new Map<string, GraphNode[]>();
  for (const ent of entityNodes) {
    entityMemoriesMap.set(ent.id, []);
  }
  const orphanMemories: GraphNode[] = [];

  for (const mem of memoryNodes) {
    const entId = memoryToEntityMap.get(mem.id);
    if (entId && entityMemoriesMap.has(entId)) {
      entityMemoriesMap.get(entId)!.push(mem);
    } else {
      orphanMemories.push(mem);
    }
  }

  const nodes: Node[] = [];
  let currentY = 50;

  for (const ent of entityNodes) {
    const startY = currentY;
    nodes.push({
      id: ent.id,
      type: 'entity',
      position: { x: 50, y: startY + 20 },
      data: { id: ent.id, label: ent.label, meta: ent.meta },
    });

    const memories = entityMemoriesMap.get(ent.id) ?? [];
    memories.forEach((mem, idx) => {
      const isHistorical =
        mem.status === 'historical' || mem.status === 'rejected';
      const col = isHistorical ? 1 : 0;
      const x = 360 + col * 320;
      const y = startY + idx * 140;

      nodes.push({
        id: mem.id,
        type: 'memory',
        position: { x, y },
        data: {
          id: mem.id,
          label: mem.label,
          status: mem.status,
          confidence: mem.confidence,
          needs_review: mem.needs_review,
          meta: mem.meta,
        },
      });
    });

    const entityHeight = Math.max(1, memories.length) * 150 + 60;
    currentY += entityHeight;
  }

  if (orphanMemories.length > 0) {
    orphanMemories.forEach((mem, idx) => {
      nodes.push({
        id: mem.id,
        type: 'memory',
        position: { x: 360, y: currentY + idx * 140 },
        data: {
          id: mem.id,
          label: mem.label,
          status: mem.status,
          confidence: mem.confidence,
          needs_review: mem.needs_review,
          meta: mem.meta,
        },
      });
    });
  }

  return nodes;
}

/**
 * Transforms backend API graph edges into styled React Flow edges.
 * Colors and dash styles reflect edge semantics (about, supersedes, refines, etc.).
 * When validNodeIds is provided, filters out dangling edges whose source or target
 * node is absent from the rendered node list.
 */
export function layoutGraphEdges(
  apiEdges: GraphEdge[] = [],
  validNodeIds?: Set<string> | Node[]
): Edge[] {
  const nodeSet =
    validNodeIds instanceof Set
      ? validNodeIds
      : Array.isArray(validNodeIds)
        ? new Set(validNodeIds.map((n) => n.id))
        : undefined;

  const validEdges = (apiEdges ?? []).filter((e) => {
    if (!nodeSet) return true;
    return nodeSet.has(e.source) && nodeSet.has(e.target);
  });

  return validEdges.map((e) => {
    let stroke = '#64748b';
    let strokeDasharray: string | undefined = undefined;

    switch (e.type) {
      case 'supersedes':
        stroke = '#c084fc';
        break;
      case 'contradicts':
        stroke = '#fbbf24';
        break;
      case 'depends_on':
        stroke = '#34d399';
        break;
      case 'refines':
        stroke = '#22d3ee';
        break;
      case 'about':
        stroke = '#64748b';
        strokeDasharray = '4 4';
        break;
    }

    return {
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.type,
      labelStyle: { fill: '#94a3b8', fontSize: 10, fontWeight: 600 },
      labelBgStyle: { fill: '#0f172a', fillOpacity: 0.8 },
      labelBgPadding: [4, 2] as [number, number],
      style: { stroke, strokeWidth: 1.5, strokeDasharray },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: stroke,
        width: 14,
        height: 14,
      },
    };
  });
}

/**
 * Unified transformer from raw backend GraphResponse to React Flow nodes and edges.
 * Guarantees that the returned edge list contains no dangling edges.
 */
export function transformGraphResponse(graph?: {
  nodes?: GraphNode[];
  edges?: GraphEdge[];
}): { nodes: Node[]; edges: Edge[] } {
  const nodes = layoutGraphNodes(graph?.nodes ?? [], graph?.edges ?? []);
  const validNodeIds = new Set(nodes.map((n) => n.id));
  const edges = layoutGraphEdges(graph?.edges ?? [], validNodeIds);
  return { nodes, edges };
}
