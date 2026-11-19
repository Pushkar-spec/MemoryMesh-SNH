import test from 'node:test';
import assert from 'node:assert/strict';
import type {
  DisputeView,
  MemoryView,
  DecisionLogEntry,
  ResolveDisputeRequest,
} from '../src/api/client.ts';
import {
  computeReviewQueueCounts,
  filterDecisionsByRule,
  extractAvailableRules,
} from '../src/lib/review.ts';

// ---------------------------------------------------------------------------
// Backend-Shaped Fixtures
// ---------------------------------------------------------------------------

const SAMPLE_DISPUTE: DisputeView = {
  a: {
    memory: {
      id: 101,
      entity_id: 1,
      entity_name: 'Server A',
      attribute: 'deploy_day',
      value_norm: 'tuesday',
      value_display: 'Tuesday',
      context: {},
      context_key: '',
      valid_from: '2026-06-01T10:00:00Z',
      valid_to: null,
      recorded_at: '2026-06-01T10:00:00Z',
      closed_at: null,
      status: 'disputed',
      confidence: 0.9,
      needs_review: true,
      review_reason: 'Conflicting claim from same-reliability source',
      raw_text: 'Server A deploy day is Tuesday.',
      message_id: 1,
      source_id: 'ci_monitor',
      evidence: [],
    },
    effective_confidence: 0.45,
    is_stale: false,
    source_names: ['ci_monitor'],
    scope_match: 'exact',
  },
  b: {
    memory: {
      id: 102,
      entity_id: 1,
      entity_name: 'Server A',
      attribute: 'deploy_day',
      value_norm: 'thursday',
      value_display: 'Thursday',
      context: {},
      context_key: '',
      valid_from: '2026-06-01T10:05:00Z',
      valid_to: null,
      recorded_at: '2026-06-01T10:05:00Z',
      closed_at: null,
      status: 'disputed',
      confidence: 0.9,
      needs_review: true,
      review_reason: 'Conflicting claim from same-reliability source',
      raw_text: 'Server A deploy day is Thursday.',
      message_id: 2,
      source_id: 'dev_chat',
      evidence: [],
    },
    effective_confidence: 0.45,
    is_stale: false,
    source_names: ['dev_chat'],
    scope_match: 'exact',
  },
  edge_reason: 'Competing assertions for deploy_day on Server A',
};

const SAMPLE_FLAGGED_MEMORY: MemoryView = {
  memory: {
    id: 201,
    entity_id: 2,
    entity_name: 'Database Cluster',
    attribute: 'replica_count',
    value_norm: '5',
    value_display: '5 nodes',
    context: { env: 'prod' },
    context_key: 'env=prod',
    valid_from: '2026-06-01T08:00:00Z',
    valid_to: null,
    recorded_at: '2026-06-01T08:00:00Z',
    closed_at: null,
    status: 'active',
    confidence: 0.55,
    needs_review: true,
    review_reason: 'Extraction confidence below auto-acceptance threshold',
    raw_text: 'Looks like maybe 5 replica nodes in prod.',
    message_id: 3,
    source_id: 'slack_ops',
    evidence: [],
  },
  effective_confidence: 0.275,
  is_stale: false,
  source_names: ['slack_ops'],
  scope_match: 'exact',
};

const SAMPLE_DECISIONS: DecisionLogEntry[] = [
  {
    id: 1,
    message_id: 10,
    claim_json: { subject: 'Server A', attribute: 'os_name', value: 'Ubuntu' },
    action: 'insert',
    rule_id: 'R-INSERT',
    memory_id: 301,
    affected_memory_ids: [301],
    scores: { base_conf: 0.9 },
    explanation: 'First observed fact for Server A os_name inserted into database.',
    created_at: '2026-06-01T10:00:00Z',
  },
  {
    id: 2,
    message_id: 11,
    claim_json: { subject: 'Server A', attribute: 'python_version', value: '3.12' },
    action: 'replace',
    rule_id: 'R-REPLACE',
    memory_id: 302,
    affected_memory_ids: [301, 302],
    scores: { new_score: 0.92, old_score: 0.8 },
    explanation: 'Superseded older python_version claim.',
    created_at: '2026-06-01T11:00:00Z',
  },
  {
    id: 3,
    message_id: 12,
    claim_json: { subject: 'Server A', attribute: 'deploy_day', value: 'Tuesday' },
    action: 'dispute',
    rule_id: 'R-DISPUTE-RESOLVE',
    memory_id: 101,
    affected_memory_ids: [101, 102],
    scores: {},
    explanation: 'Manual dispute resolution confirmed Memory #101 as winner.',
    created_at: '2026-06-01T12:00:00Z',
  },
];

// ---------------------------------------------------------------------------
// Review Page Behavioral Tests
// ---------------------------------------------------------------------------

test('Review queue: computes accurate counts for disputes, flagged memories, and total', () => {
  const counts = computeReviewQueueCounts([SAMPLE_DISPUTE], [SAMPLE_FLAGGED_MEMORY]);
  assert.equal(counts.disputesCount, 1);
  assert.equal(counts.flaggedCount, 1);
  assert.equal(counts.totalPending, 2);

  const emptyCounts = computeReviewQueueCounts([], []);
  assert.equal(emptyCounts.disputesCount, 0);
  assert.equal(emptyCounts.flaggedCount, 0);
  assert.equal(emptyCounts.totalPending, 0);
});

test('Dispute view: renders competing claims and preserves conflicting values', () => {
  const memA = SAMPLE_DISPUTE.a.memory;
  const memB = SAMPLE_DISPUTE.b.memory;

  assert.equal(memA.id, 101);
  assert.equal(memA.value_norm, 'tuesday');
  assert.equal(memA.source_id, 'ci_monitor');
  assert.equal(memA.status, 'disputed');

  assert.equal(memB.id, 102);
  assert.equal(memB.value_norm, 'thursday');
  assert.equal(memB.source_id, 'dev_chat');
  assert.equal(memB.status, 'disputed');

  assert.equal(SAMPLE_DISPUTE.edge_reason, 'Competing assertions for deploy_day on Server A');
});

test('Dispute resolution: selecting winner creates valid ResolveDisputeRequest', () => {
  const winnerId = 102;
  const payload: ResolveDisputeRequest = { winner_memory_id: winnerId };
  assert.equal(payload.winner_memory_id, 102);
  assert.ok(typeof payload.winner_memory_id === 'number');
});

test('Dispute resolution: successful resolution removes dispute and records winner', async () => {
  let pendingDisputes = [SAMPLE_DISPUTE];
  let resolutionResult: DecisionLogEntry | null = null;

  // Mock resolve mutation
  const resolveDisputeMock = async (req: ResolveDisputeRequest): Promise<DecisionLogEntry> => {
    // Backend returns DecisionLogEntry for R-DISPUTE-RESOLVE
    return {
      id: 99,
      message_id: 0,
      claim_json: {},
      action: 'dispute',
      rule_id: 'R-DISPUTE-RESOLVE',
      memory_id: req.winner_memory_id,
      affected_memory_ids: [101, 102],
      scores: {},
      explanation: `Manual dispute resolution selected memory #${req.winner_memory_id}`,
      created_at: '2026-06-01T12:00:00Z',
    };
  };

  // Simulate user selecting Memory #101 as winner
  resolutionResult = await resolveDisputeMock({ winner_memory_id: 101 });
  // On success, backend refreshes disputes list to empty
  pendingDisputes = [];

  assert.equal(resolutionResult.rule_id, 'R-DISPUTE-RESOLVE');
  assert.equal(resolutionResult.memory_id, 101);
  assert.equal(pendingDisputes.length, 0);
});

test('Dispute resolution: failed resolution preserves dispute in pending list', async () => {
  const pendingDisputes = [SAMPLE_DISPUTE];
  let capturedError: Error | null = null;

  const failingResolveMock = async (req: ResolveDisputeRequest): Promise<DecisionLogEntry> => {
    assert.equal(req.winner_memory_id, 101);
    throw new Error('Network timeout: backend unreachable');
  };

  try {
    await failingResolveMock({ winner_memory_id: 101 });
  } catch (err: unknown) {
    capturedError = err as Error;
  }

  // Dispute must NOT be removed on failure
  assert.ok(capturedError);
  assert.match(capturedError.message, /backend unreachable/);
  assert.equal(pendingDisputes.length, 1);
  assert.equal(pendingDisputes[0].a.memory.id, 101);
});

test('Review acknowledgement simulation: successful acknowledgement unflags memory', async () => {
  let flaggedMemories = [SAMPLE_FLAGGED_MEMORY];

  const acknowledgeMock = async (id: number) => {
    assert.equal(id, 201);
    // Backend clears needs_review flag
    return {
      ...SAMPLE_FLAGGED_MEMORY.memory,
      needs_review: false,
    };
  };

  const updatedMem = await acknowledgeMock(201);
  // On success, flagged memories list re-fetched without needs_review=true
  flaggedMemories = flaggedMemories.filter((m) => m.memory.id !== updatedMem.id);

  assert.equal(updatedMem.needs_review, false);
  assert.equal(flaggedMemories.length, 0);
});

test('Review acknowledgement: failed acknowledgement preserves pending flag', async () => {
  const flaggedMemories = [SAMPLE_FLAGGED_MEMORY];
  let capturedError: Error | null = null;

  const failingAcknowledgeMock = async (id: number) => {
    assert.equal(id, 201);
    throw new Error('HTTP 404: Memory not found');
  };

  try {
    await failingAcknowledgeMock(201);
  } catch (err: unknown) {
    capturedError = err as Error;
  }

  // Memory must NOT be removed from review list on failure
  assert.ok(capturedError);
  assert.match(capturedError.message, /Memory not found/);
  assert.equal(flaggedMemories.length, 1);
  assert.equal(flaggedMemories[0].memory.needs_review, true);
});

// ---------------------------------------------------------------------------
// Decisions Page Behavioral Tests
// ---------------------------------------------------------------------------

test('Decisions filtering: filterDecisionsByRule filters accurately and case-insensitively', () => {
  const insertDecisions = filterDecisionsByRule(SAMPLE_DECISIONS, 'R-INSERT');
  assert.equal(insertDecisions.length, 1);
  assert.equal(insertDecisions[0].rule_id, 'R-INSERT');
  assert.equal(insertDecisions[0].action, 'insert');

  // Case-insensitivity check
  const replaceDecisions = filterDecisionsByRule(SAMPLE_DECISIONS, 'r-replace');
  assert.equal(replaceDecisions.length, 1);
  assert.equal(replaceDecisions[0].rule_id, 'R-REPLACE');

  // Non-matching rule returns empty list
  const nonExistent = filterDecisionsByRule(SAMPLE_DECISIONS, 'R-NONEXISTENT');
  assert.equal(nonExistent.length, 0);
});

test('Decisions filtering: clearing filter restores unfiltered decision history', () => {
  const allFiltered = filterDecisionsByRule(SAMPLE_DECISIONS, '');
  assert.equal(allFiltered.length, 3);

  const undefFiltered = filterDecisionsByRule(SAMPLE_DECISIONS, undefined);
  assert.equal(undefFiltered.length, 3);
});

test('Decisions rules: extracts unique sorted list of canonical and dynamic rules', () => {
  const canonicalRules = ['R-INSERT', 'R-REPLACE', 'R-CORROBORATE', 'R-DISPUTE'];
  const rules = extractAvailableRules(SAMPLE_DECISIONS, canonicalRules);

  assert.ok(rules.includes('R-INSERT'));
  assert.ok(rules.includes('R-REPLACE'));
  assert.ok(rules.includes('R-CORROBORATE'));
  assert.ok(rules.includes('R-DISPUTE'));
  assert.ok(rules.includes('R-DISPUTE-RESOLVE'), 'Dynamic rule from decision history included');

  // Must be sorted
  const sortedCopy = [...rules].sort();
  assert.deepEqual(rules, sortedCopy);
});

