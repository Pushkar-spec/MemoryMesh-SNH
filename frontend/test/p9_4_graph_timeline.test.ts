import test from 'node:test';
import assert from 'node:assert/strict';
import type { GraphNode, GraphEdge, SlotTimeline } from '../src/api/client.ts';
import {
  layoutGraphNodes,
  layoutGraphEdges,
  transformGraphResponse,
} from '../src/lib/graphLayout.ts';
import { getTimelineLaneKey, getTimelineLaneStats } from '../src/lib/timeline.ts';
import { formatValidityInterval, formatUtcDateTime, formatConfidence } from '../src/lib/format.ts';

// ---------------------------------------------------------------------------
// Backend-Shaped Fixtures
// ---------------------------------------------------------------------------

const BACKEND_GRAPH_FIXTURE: { nodes: GraphNode[]; edges: GraphEdge[] } = {
  nodes: [
    {
      id: 'entity:1',
      kind: 'entity',
      label: 'Server A',
      status: null,
      confidence: null,
      needs_review: false,
      meta: { name: 'Server A', id: 1 },
    },
    {
      id: 'mem:1',
      kind: 'memory',
      label: 'python_version: 3.11',
      status: 'historical',
      confidence: 0.81,
      needs_review: false,
      meta: {
        value: '3.11',
        valid_from: '2026-06-01T10:00:00Z',
        valid_to: '2026-06-01T11:00:00Z',
        source_id: 'sysadmin_priya',
        entity: 'Server A',
        attribute: 'python_version',
        context_key: '',
      },
    },
    {
      id: 'mem:2',
      kind: 'memory',
      label: 'python_version: 3.12',
      status: 'active',
      confidence: 0.9,
      needs_review: true,
      meta: {
        value: '3.12',
        valid_from: '2026-06-01T11:00:00Z',
        valid_to: null,
        source_id: 'sysadmin_priya',
        entity: 'Server A',
        attribute: 'python_version',
        context_key: '',
      },
    },
  ],
  edges: [
    {
      id: 'about:1->1',
      source: 'mem:1',
      target: 'entity:1',
      type: 'about',
      reason: 'Memory 1 is about entity Server A',
    },
    {
      id: 'about:2->1',
      source: 'mem:2',
      target: 'entity:1',
      type: 'about',
      reason: 'Memory 2 is about entity Server A',
    },
    {
      id: 'edge:1',
      source: 'mem:2',
      target: 'mem:1',
      type: 'supersedes',
      reason: 'Server migrated from 3.11 to 3.12',
    },
  ],
};

// ---------------------------------------------------------------------------
// P9.4 Production Graph Transformation & Layout Tests
// ---------------------------------------------------------------------------

test('Production layoutGraphNodes: maps entity and memory nodes with DAG columns', () => {
  const nodes = layoutGraphNodes(
    BACKEND_GRAPH_FIXTURE.nodes,
    BACKEND_GRAPH_FIXTURE.edges
  );

  assert.equal(nodes.length, 3);

  // 1. Entity Node validation
  const entNode = nodes.find((n) => n.id === 'entity:1');
  assert.ok(entNode, 'Entity node entity:1 must exist in layout');
  assert.equal(entNode.type, 'entity');
  assert.equal(entNode.position.x, 50, 'Entity nodes must be placed at x=50');
  assert.deepEqual(entNode.data, {
    id: 'entity:1',
    label: 'Server A',
    meta: { name: 'Server A', id: 1 },
  });

  // 2. Active Memory Node validation (placed in primary active column x=360)
  const activeNode = nodes.find((n) => n.id === 'mem:2');
  assert.ok(activeNode, 'Memory node mem:2 must exist in layout');
  assert.equal(activeNode.type, 'memory');
  assert.equal(activeNode.position.x, 360, 'Active memory placed at x=360');
  assert.equal(activeNode.data.status, 'active');
  assert.equal(activeNode.data.confidence, 0.9);
  assert.equal(activeNode.data.needs_review, true);

  // 3. Historical Memory Node validation (placed in revision column x=680)
  const histNode = nodes.find((n) => n.id === 'mem:1');
  assert.ok(histNode, 'Memory node mem:1 must exist in layout');
  assert.equal(histNode.type, 'memory');
  assert.equal(histNode.position.x, 680, 'Historical memory placed at x=680');
  assert.equal(histNode.data.status, 'historical');
  assert.equal(histNode.data.confidence, 0.81);
  assert.equal(histNode.data.needs_review, false);
});

test('Production layoutGraphEdges: preserves memory -> entity direction and edge styles', () => {
  const edges = layoutGraphEdges(BACKEND_GRAPH_FIXTURE.edges);

  assert.equal(edges.length, 3);

  // 1. Verify "about" edge preserves memory -> entity direction
  const aboutEdges = edges.filter((e) => e.label === 'about');
  assert.equal(aboutEdges.length, 2);
  for (const edge of aboutEdges) {
    assert.match(edge.source, /^mem:\d+$/, 'about edge source must be a memory');
    assert.match(edge.target, /^entity:\d+$/, 'about edge target must be an entity');
    assert.equal(edge.style?.strokeDasharray, '4 4', 'about edge must be dashed');
    assert.equal(edge.style?.stroke, '#64748b', 'about edge stroke must be slate-500');
  }

  // 2. Verify "supersedes" relationship edge preserves source -> target
  const supersedesEdge = edges.find((e) => e.label === 'supersedes');
  assert.ok(supersedesEdge, 'supersedes edge must exist');
  assert.equal(supersedesEdge.id, 'edge:1');
  assert.equal(supersedesEdge.source, 'mem:2', 'source is superseding memory');
  assert.equal(supersedesEdge.target, 'mem:1', 'target is superseded memory');
  assert.equal(supersedesEdge.style?.stroke, '#c084fc', 'supersedes edge is purple');
  assert.ok(supersedesEdge.markerEnd, 'markerEnd arrow must be present');
});

test('Production layoutGraphEdges: handles all relationship edge types and styles', () => {
  const relationEdges: GraphEdge[] = [
    { id: 'e-sup', source: 'mem:2', target: 'mem:1', type: 'supersedes', reason: 'r1' },
    { id: 'e-ref', source: 'mem:3', target: 'mem:1', type: 'refines', reason: 'r2' },
    { id: 'e-con', source: 'mem:4', target: 'mem:5', type: 'contradicts', reason: 'r3' },
    { id: 'e-dep', source: 'mem:6', target: 'mem:7', type: 'depends_on', reason: 'r4' },
    { id: 'e-abt', source: 'mem:2', target: 'entity:1', type: 'about', reason: 'r5' },
  ];

  const edges = layoutGraphEdges(relationEdges);
  assert.equal(edges.length, 5);

  const styleMap = Object.fromEntries(
    edges.map((e) => [
      e.label,
      { stroke: e.style?.stroke, dashed: Boolean(e.style?.strokeDasharray) },
    ])
  );

  assert.deepEqual(styleMap['supersedes'], { stroke: '#c084fc', dashed: false });
  assert.deepEqual(styleMap['refines'], { stroke: '#22d3ee', dashed: false });
  assert.deepEqual(styleMap['contradicts'], { stroke: '#fbbf24', dashed: false });
  assert.deepEqual(styleMap['depends_on'], { stroke: '#34d399', dashed: false });
  assert.deepEqual(styleMap['about'], { stroke: '#64748b', dashed: true });
});

test('Production graph transformation: filters edges with missing source or missing target', () => {
  const nodes: GraphNode[] = [
    { id: 'entity:1', kind: 'entity', label: 'Server A', meta: {} },
    { id: 'mem:1', kind: 'memory', label: 'python_version: 3.11', status: 'active', meta: {} },
    { id: 'mem:2', kind: 'memory', label: 'python_version: 3.12', status: 'active', meta: {} },
  ];

  const edgesWithDangling: GraphEdge[] = [
    // 1. Valid edge: both source and target exist
    { id: 'valid-about', source: 'mem:1', target: 'entity:1', type: 'about' },
    { id: 'valid-rel', source: 'mem:2', target: 'mem:1', type: 'supersedes' },
    // 2. Dangling edge with missing source
    { id: 'dangling-src', source: 'mem:nonexistent_src', target: 'entity:1', type: 'about' },
    // 3. Dangling edge with missing target
    { id: 'dangling-tgt', source: 'mem:2', target: 'entity:nonexistent_tgt', type: 'about' },
    // 4. Dangling edge where both endpoints are missing
    { id: 'dangling-both', source: 'mem:ghost1', target: 'mem:ghost2', type: 'depends_on' },
  ];

  const result = transformGraphResponse({ nodes, edges: edgesWithDangling });

  // Rendered node count must be 3
  assert.equal(result.nodes.length, 3);

  // Exactly the 2 valid edges must be preserved; all 3 dangling edges must be excluded
  assert.equal(result.edges.length, 2);
  const resultEdgeIds = result.edges.map((e) => e.id);
  assert.ok(resultEdgeIds.includes('valid-about'));
  assert.ok(resultEdgeIds.includes('valid-rel'));
  assert.ok(!resultEdgeIds.includes('dangling-src'), 'Edge with missing source must be filtered');
  assert.ok(!resultEdgeIds.includes('dangling-tgt'), 'Edge with missing target must be filtered');
  assert.ok(
    !resultEdgeIds.includes('dangling-both'),
    'Edge with missing endpoints must be filtered'
  );

  // Verify layoutGraphEdges directly with node list
  const directFiltered = layoutGraphEdges(edgesWithDangling, result.nodes);
  assert.equal(directFiltered.length, 2);
});

test('Production layoutGraphNodes: safely handles orphan memories without throwing', () => {
  const unlinkedNodes: GraphNode[] = [
    { id: 'entity:1', kind: 'entity', label: 'Server A', meta: {} },
    { id: 'mem:10', kind: 'memory', label: 'status: online', status: 'active', meta: {} },
    { id: 'mem:99', kind: 'memory', label: 'orphan_attr: val', status: 'active', meta: {} },
  ];

  // mem:10 links to entity:1, but mem:99 links to nonexistent entity:999
  const unlinkedEdges: GraphEdge[] = [
    { id: 'a1', source: 'mem:10', target: 'entity:1', type: 'about' },
    { id: 'a2', source: 'mem:99', target: 'entity:999', type: 'about' },
  ];

  const nodes = layoutGraphNodes(unlinkedNodes, unlinkedEdges);
  assert.equal(nodes.length, 3);

  const orphanNode = nodes.find((n) => n.id === 'mem:99');
  assert.ok(orphanNode, 'Orphan memory node must be preserved in layout');
  assert.equal(orphanNode.type, 'memory');
  assert.equal(orphanNode.position.x, 360);
});

test('Production transformGraphResponse: safely handles empty and undefined graphs', () => {
  const emptyRes = transformGraphResponse({ nodes: [], edges: [] });
  assert.deepEqual(emptyRes.nodes, []);
  assert.deepEqual(emptyRes.edges, []);

  const undefRes = transformGraphResponse(undefined);
  assert.deepEqual(undefRes.nodes, []);
  assert.deepEqual(undefRes.edges, []);

  const partialRes = transformGraphResponse({});
  assert.deepEqual(partialRes.nodes, []);
  assert.deepEqual(partialRes.edges, []);
});

// ---------------------------------------------------------------------------
// P9.4 Production Timeline Transformation Tests
// ---------------------------------------------------------------------------

test('Production getTimelineLaneKey: generates collision-free composite keys', () => {
  // Scenario that would collide under naive hyphen concatenation:
  // Lane 1: entity="Server-A", attribute="os", context="" -> "Server-A-os-"
  // Lane 2: entity="Server", attribute="A-os", context="" -> "Server-A-os-"
  const lane1: SlotTimeline = {
    entity: 'Server-A',
    attribute: 'os',
    context_key: '',
  };
  const lane2: SlotTimeline = {
    entity: 'Server',
    attribute: 'A-os',
    context_key: '',
  };

  const key1 = getTimelineLaneKey(lane1);
  const key2 = getTimelineLaneKey(lane2);

  assert.notEqual(key1, key2, 'Distinct entity/attribute pairs must not collide');
  assert.equal(key1, JSON.stringify(['Server-A', 'os', '']));
  assert.equal(key2, JSON.stringify(['Server', 'A-os', '']));

  // Context variation collision resistance
  const laneCtxA: SlotTimeline = {
    entity: 'Server',
    attribute: 'python',
    context_key: 'version-3',
  };
  const laneCtxB: SlotTimeline = {
    entity: 'Server',
    attribute: 'python-version',
    context_key: '3',
  };

  assert.notEqual(
    getTimelineLaneKey(laneCtxA),
    getTimelineLaneKey(laneCtxB),
    'Lanes with boundary context hyphens must produce distinct composite keys'
  );

  // Idempotency: identical data produces identical keys
  assert.equal(getTimelineLaneKey(lane1), getTimelineLaneKey({ ...lane1 }));
});

test('Production getTimelineLaneStats: separates all 4 memory statuses', () => {
  const laneWithAllStatuses: SlotTimeline = {
    entity: 'Server A',
    attribute: 'python_version',
    context_key: '',
    memories: [
      // 1. Historical Memory
      {
        memory: {
          id: 1,
          entity_id: 1,
          entity_name: 'Server A',
          attribute: 'python_version',
          value_norm: '3.10',
          value_display: '3.10',
          context: {},
          context_key: '',
          valid_from: '2026-06-01T09:00:00Z',
          valid_to: '2026-06-01T10:00:00Z',
          recorded_at: '2026-06-01T09:00:00Z',
          closed_at: '2026-06-01T10:00:00Z',
          status: 'historical',
          confidence: 0.8,
          needs_review: false,
          review_reason: '',
          raw_text: '',
          message_id: 1,
          source_id: 'sysadmin',
          evidence: [],
        },
        effective_confidence: 0.8,
        is_stale: false,
        source_names: ['sysadmin'],
        scope_match: 'exact',
      },
      // 2. Active Memory
      {
        memory: {
          id: 2,
          entity_id: 1,
          entity_name: 'Server A',
          attribute: 'python_version',
          value_norm: '3.12',
          value_display: '3.12',
          context: {},
          context_key: '',
          valid_from: '2026-06-01T11:00:00Z',
          valid_to: null,
          recorded_at: '2026-06-01T11:00:00Z',
          closed_at: null,
          status: 'active',
          confidence: 0.95,
          needs_review: false,
          review_reason: '',
          raw_text: '',
          message_id: 2,
          source_id: 'sysadmin',
          evidence: [],
        },
        effective_confidence: 0.95,
        is_stale: false,
        source_names: ['sysadmin'],
        scope_match: 'exact',
      },
      // 3. Disputed Memory (must NOT be counted as historical)
      {
        memory: {
          id: 3,
          entity_id: 1,
          entity_name: 'Server A',
          attribute: 'python_version',
          value_norm: '3.11',
          value_display: '3.11',
          context: {},
          context_key: '',
          valid_from: '2026-06-01T10:00:00Z',
          valid_to: null,
          recorded_at: '2026-06-01T10:00:00Z',
          closed_at: null,
          status: 'disputed',
          confidence: 0.5,
          needs_review: true,
          review_reason: 'Conflicting upgrade claim',
          raw_text: '',
          message_id: 3,
          source_id: 'contractor',
          evidence: [],
        },
        effective_confidence: 0.25,
        is_stale: false,
        source_names: ['contractor'],
        scope_match: 'exact',
      },
    ],
    // 4. Rejected Memory (in backend rejected array)
    rejected: [
      {
        memory: {
          id: 4,
          entity_id: 1,
          entity_name: 'Server A',
          attribute: 'python_version',
          value_norm: '2.7',
          value_display: '2.7',
          context: {},
          context_key: '',
          valid_from: '2026-06-01T08:00:00Z',
          valid_to: null,
          recorded_at: '2026-06-01T08:00:00Z',
          closed_at: null,
          status: 'rejected',
          confidence: 0.1,
          needs_review: true,
          review_reason: 'Contradicted by authoritative source',
          raw_text: '',
          message_id: 4,
          source_id: 'untrusted_source',
          evidence: [],
        },
        effective_confidence: 0.0,
        is_stale: false,
        source_names: ['untrusted_source'],
        scope_match: 'exact',
      },
    ],
  };

  const stats = getTimelineLaneStats(laneWithAllStatuses);

  assert.equal(stats.activeCount, 1, 'activeCount must only count status=active');
  assert.equal(stats.historicalCount, 1, 'historicalCount must only count status=historical');
  assert.equal(stats.disputedCount, 1, 'disputedCount must only count status=disputed');
  assert.equal(stats.rejectedCount, 1, 'rejectedCount must count rejected claims array');
  assert.equal(stats.totalCount, 4, 'totalCount must be sum of memories and rejected');
});

test('Timeline Formatting: correctly presents bitemporal intervals and confidence', () => {
  const activeInterval = formatValidityInterval('2026-06-01T11:00:00Z', null);
  assert.equal(activeInterval, '2026-06-01 → Present');

  const closedInterval = formatValidityInterval(
    '2026-06-01T10:00:00Z',
    '2026-06-01T11:00:00Z'
  );
  assert.equal(closedInterval, '2026-06-01 → 2026-06-01');

  assert.match(
    formatUtcDateTime('2026-06-01T11:00:00Z'),
    /2026-06-01 11:00:00 UTC/
  );
  assert.equal(formatConfidence(0.8123), '81.2%');
  assert.equal(formatConfidence(null), '—');
});
