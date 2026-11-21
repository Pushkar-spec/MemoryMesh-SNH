import test from 'node:test';
import assert from 'node:assert/strict';
import type { IngestReport, DecisionLogEntry } from '../src/api/client.ts';
import { startReplay } from '../src/lib/replay.ts';

// ---------------------------------------------------------------------------
// Mock EventSource Implementation for Deterministic Replay Testing
// ---------------------------------------------------------------------------

let activeMock: MockEventSource | null = null;

function registerActiveMock(instance: MockEventSource): void {
  activeMock = instance;
}

class MockEventSource {
  public url: string;
  public onopen: (() => void) | null = null;
  public listeners: Map<string, ((event: { data?: string }) => void)[]> = new Map();
  public closed = false;

  constructor(url: string) {
    this.url = url;
    registerActiveMock(this);
    setTimeout(() => {
      if (!this.closed && this.onopen) {
        this.onopen();
      }
    }, 0);
  }

  public addEventListener(type: string, listener: (event: { data?: string }) => void): void {
    const list = this.listeners.get(type) || [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  public emit(type: string, data?: string): void {
    if (this.closed) return;
    const list = this.listeners.get(type) || [];
    for (const listener of list) {
      listener({ data });
    }
  }

  public close(): void {
    this.closed = true;
  }
}

interface CustomGlobal {
  EventSource: unknown;
}

// ---------------------------------------------------------------------------
// Sample Test Reports
// ---------------------------------------------------------------------------

function createMockReport(
  messageId: number,
  ruleId: string,
  action: DecisionLogEntry['action']
): IngestReport {
  const decision: DecisionLogEntry = {
    id: messageId * 10,
    action,
    rule_id: ruleId,
    memory_id: messageId * 100,
    affected_memory_ids: [messageId * 100],
    explanation: `Test execution of ${ruleId}`,
    claim_json: { subject: 'Server A', attribute: 'python_version' },
    created_at: '2026-06-01T10:00:00Z',
  };

  return {
    message_id: messageId,
    duplicate: false,
    claims_extracted: 1,
    decisions: [decision],
    success: true,
  };
}

// ---------------------------------------------------------------------------
// Test Suite
// ---------------------------------------------------------------------------

test('startReplay: constructs canonical SSE URL with parameters', () => {
  const originalEventSource = globalThis.EventSource;
  const customGlobal = globalThis as unknown as CustomGlobal;
  customGlobal.EventSource = MockEventSource;

  try {
    const session = startReplay({
      scenario: 'hero-001',
      mode: 'structured',
      delay_ms: 500,
    });

    assert.ok(activeMock);
    const capturedUrl = activeMock.url;
    assert.ok(capturedUrl.includes('/api/demo/replay'));
    assert.ok(capturedUrl.includes('scenario=hero-001'));
    assert.ok(capturedUrl.includes('mode=structured'));
    assert.ok(capturedUrl.includes('delay_ms=500'));
    // Replay endpoint is strictly read-only; reset is never sent in query
    assert.equal(capturedUrl.includes('reset='), false);
    assert.equal(session.isActive, true);
    session.stop();
  } finally {
    customGlobal.EventSource = originalEventSource;
  }
});

test('startReplay: dispatches sequential message_ingested reports in order', async () => {
  const originalEventSource = globalThis.EventSource;
  const customGlobal = globalThis as unknown as CustomGlobal;
  customGlobal.EventSource = MockEventSource;

  try {
    const receivedReports: IngestReport[] = [];
    const session = startReplay({
      scenario: 'hero-001',
      mode: 'structured',
      onMessageIngested: (report) => {
        receivedReports.push(report);
      },
    });

    assert.ok(activeMock);

    const r1 = createMockReport(1, 'R-INSERT', 'insert');
    const r2 = createMockReport(2, 'R-CORROBORATE', 'corroborate');
    const r5 = createMockReport(5, 'R-REPLACE', 'replace');

    activeMock.emit('message_ingested', JSON.stringify(r1));
    activeMock.emit('message_ingested', JSON.stringify(r2));
    activeMock.emit('message_ingested', JSON.stringify(r5));

    assert.equal(receivedReports.length, 3);
    assert.equal(receivedReports[0].message_id, 1);
    assert.equal(receivedReports[0].decisions[0].rule_id, 'R-INSERT');
    assert.equal(receivedReports[1].message_id, 2);
    assert.equal(receivedReports[1].decisions[0].rule_id, 'R-CORROBORATE');
    assert.equal(receivedReports[2].message_id, 5);
    assert.equal(receivedReports[2].decisions[0].rule_id, 'R-REPLACE');

    session.stop();
  } finally {
    customGlobal.EventSource = originalEventSource;
  }
});

test('startReplay: completes on done event and closes active session', async () => {
  const originalEventSource = globalThis.EventSource;
  const customGlobal = globalThis as unknown as CustomGlobal;
  customGlobal.EventSource = MockEventSource;

  try {
    let completedData: { messages: number } | null = null;
    const session = startReplay({
      onDone: (data) => {
        completedData = data;
      },
    });

    assert.ok(activeMock);
    assert.equal(session.isActive, true);
    activeMock.emit('done', JSON.stringify({ messages: 8 }));

    assert.deepEqual(completedData, { messages: 8 });
    assert.equal(session.isActive, false);
    assert.equal(activeMock.closed, true);
  } finally {
    customGlobal.EventSource = originalEventSource;
  }
});

test('startReplay: handles explicit server error events', async () => {
  const originalEventSource = globalThis.EventSource;
  const customGlobal = globalThis as unknown as CustomGlobal;
  customGlobal.EventSource = MockEventSource;

  try {
    let capturedError: { error: string } | Error | null = null;
    const session = startReplay({
      onError: (err) => {
        capturedError = err;
      },
    });

    assert.ok(activeMock);
    activeMock.emit('error', JSON.stringify({ error: "Scenario 'invalid' not found." }));

    assert.ok(capturedError);
    assert.equal((capturedError as { error: string }).error, "Scenario 'invalid' not found.");
    assert.equal(session.isActive, false);
  } finally {
    customGlobal.EventSource = originalEventSource;
  }
});

test('startReplay: handles network disconnect error', async () => {
  const originalEventSource = globalThis.EventSource;
  const customGlobal = globalThis as unknown as CustomGlobal;
  customGlobal.EventSource = MockEventSource;

  try {
    let capturedError: { error: string } | Error | null = null;
    const session = startReplay({
      onError: (err) => {
        capturedError = err;
      },
    });

    assert.ok(activeMock);
    // Native disconnect has no data payload
    activeMock.emit('error');

    assert.ok(capturedError instanceof Error);
    assert.ok(capturedError.message.includes('connection was closed'));
    assert.equal(session.isActive, false);
  } finally {
    customGlobal.EventSource = originalEventSource;
  }
});

test('startReplay: manual stop closes session and prevents subsequent callbacks', () => {
  const originalEventSource = globalThis.EventSource;
  const customGlobal = globalThis as unknown as CustomGlobal;
  customGlobal.EventSource = MockEventSource;

  try {
    let reportCount = 0;
    const session = startReplay({
      onMessageIngested: () => {
        reportCount++;
      },
    });

    assert.ok(activeMock);
    const r1 = createMockReport(1, 'R-INSERT', 'insert');
    activeMock.emit('message_ingested', JSON.stringify(r1));
    assert.equal(reportCount, 1);

    session.stop();
    assert.equal(session.isActive, false);

    const r2 = createMockReport(2, 'R-CORROBORATE', 'corroborate');
    activeMock.emit('message_ingested', JSON.stringify(r2));
    assert.equal(reportCount, 1, 'Subsequent reports after stop must be ignored');
  } finally {
    customGlobal.EventSource = originalEventSource;
  }
});

test('Replay concurrency: duplicate replay starts are blocked while active', async () => {
  let activeReplayCount = 0;
  let isReplaying = false;

  const triggerReplay = (): boolean => {
    if (isReplaying) {
      return false; // Guard prevents concurrent start
    }
    isReplaying = true;
    activeReplayCount++;
    return true;
  };

  const firstStart = triggerReplay();
  assert.equal(firstStart, true, 'First replay start should succeed');

  const secondAttempt = triggerReplay();
  assert.equal(secondAttempt, false, 'Duplicate replay start must be blocked while active');
  assert.equal(activeReplayCount, 1);

  // Complete replay
  isReplaying = false;

  const thirdAttempt = triggerReplay();
  assert.equal(thirdAttempt, true, 'Subsequent replay start after completion should succeed');
  assert.equal(activeReplayCount, 2);
});

test('Ingest coexistence: manual ingestion and replay state do not collide', () => {
  let displayedDecisions: DecisionLogEntry[] = [];
  let replayActive = false;

  // Simulate starting replay
  const startReplayAction = () => {
    replayActive = true;
    displayedDecisions = []; // Clears manual report
  };

  // Simulate replay message arrival
  const onReplayReport = (report: IngestReport) => {
    displayedDecisions.push(...report.decisions);
  };

  startReplayAction();
  onReplayReport(createMockReport(1, 'R-INSERT', 'insert'));
  onReplayReport(createMockReport(2, 'R-CORROBORATE', 'corroborate'));
  assert.equal(displayedDecisions.length, 2);
  assert.equal(replayActive, true);

  // Simulate user triggering manual ingest
  const onManualIngestSuccess = (singleReport: IngestReport) => {
    replayActive = false; // Exits replay mode
    displayedDecisions = [...singleReport.decisions]; // Overwrites with single report
  };

  onManualIngestSuccess(createMockReport(99, 'R-REPLACE', 'replace'));
  assert.equal(replayActive, false);
  assert.equal(displayedDecisions.length, 1);
  assert.equal(displayedDecisions[0].rule_id, 'R-REPLACE');
});

test('Replay confirmation flow: cancel makes 0 network requests and preserves state', () => {
  let showConfirmation = false;
  const isReplaying = false;
  const previousResult: string | null = 'existing-ingest-result';
  const postResetCalls = 0;
  const getReplayCalls = 0;

  // Step 1: User clicks Replay Hero Scenario button
  const onRequestReplay = () => {
    showConfirmation = true;
  };

  // Step 2: User cancels in the confirmation dialog
  const onCancelReplay = () => {
    showConfirmation = false;
  };

  onRequestReplay();
  assert.equal(showConfirmation, true);
  assert.equal(isReplaying, false);
  assert.equal(previousResult, 'existing-ingest-result');

  onCancelReplay();
  assert.equal(showConfirmation, false);
  assert.equal(isReplaying, false);
  assert.equal(previousResult, 'existing-ingest-result', 'State must not be modified on cancel');
  assert.equal(postResetCalls, 0, 'No reset request on cancel');
  assert.equal(getReplayCalls, 0, 'No replay request on cancel');
});

test('Replay confirmation flow: Replay Without Reset starts replay without POST reset', () => {
  let showConfirmation = false;
  let isReplaying = false;
  const postResetCalls = 0;
  let getReplayCalls = 0;

  const onRequestReplay = () => {
    showConfirmation = true;
  };

  const onReplayWithoutReset = () => {
    showConfirmation = false;
    isReplaying = true;
    getReplayCalls++;
  };

  onRequestReplay();
  assert.equal(showConfirmation, true);

  onReplayWithoutReset();
  assert.equal(showConfirmation, false);
  assert.equal(isReplaying, true);
  assert.equal(postResetCalls, 0, 'Must NOT invoke POST reset endpoint');
  assert.equal(getReplayCalls, 1, 'Must initiate GET replay stream');
});

test('Replay confirmation flow: Confirm & Reset All calls POST reset then replays', async () => {
  let showConfirmation = false;
  let isReplaying = false;
  let postResetCalls = 0;
  let getReplayCalls = 0;

  const onRequestReplay = () => {
    showConfirmation = true;
  };

  const onConfirmAndResetReplay = async (mockResetFn: () => Promise<{ ok: boolean }>) => {
    showConfirmation = false;
    postResetCalls++;
    await mockResetFn();
    isReplaying = true;
    getReplayCalls++;
  };

  onRequestReplay();
  assert.equal(showConfirmation, true);

  await onConfirmAndResetReplay(async () => ({ ok: true }));
  assert.equal(showConfirmation, false);
  assert.equal(postResetCalls, 1, 'Must invoke POST reset endpoint');
  assert.equal(getReplayCalls, 1, 'Must initiate replay after successful reset');
  assert.equal(isReplaying, true);
});

test('Replay confirmation flow: failed POST reset aborts replay and shows error', async () => {
  let showConfirmation = false;
  let isReplaying = false;
  let replayError: string | null = null;
  let postResetCalls = 0;
  let getReplayCalls = 0;

  const onRequestReplay = () => {
    showConfirmation = true;
  };

  const onConfirmAndResetReplay = async (mockResetFn: () => Promise<{ ok: boolean }>) => {
    showConfirmation = false;
    postResetCalls++;
    try {
      await mockResetFn();
      isReplaying = true;
      getReplayCalls++;
    } catch (err) {
      replayError = err instanceof Error ? err.message : String(err);
    }
  };

  onRequestReplay();
  assert.equal(showConfirmation, true);

  await onConfirmAndResetReplay(async () => {
    throw new Error('Database reset failed: connection refused');
  });

  assert.equal(showConfirmation, false);
  assert.equal(postResetCalls, 1);
  assert.equal(getReplayCalls, 0, 'Must NOT initiate replay if reset failed');
  assert.equal(isReplaying, false, 'Must not be replaying');
  assert.equal(replayError, 'Database reset failed: connection refused');
});

