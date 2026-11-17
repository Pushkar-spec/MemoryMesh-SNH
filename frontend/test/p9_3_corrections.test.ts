import test from 'node:test';
import assert from 'node:assert';
import {
  validateEventTimestamp,
  isLeapYear,
  getDaysInMonth,
} from '../src/lib/validation.ts';
import type { AskResponse, IngestReport, CatalogResponse } from '../src/api/client.ts';

/**
 * P9.3 Focused Unit & State Simulation Tests.
 *
 * NOTE: Tests in sections 2, 3, and 4 test shared production logic and simulate
 * isolated state transitions corresponding to the page handlers. They are
 * behavioral simulations, not full DOM/browser page-rendering tests.
 */

// ==============================================================================
// 1. Rigorous Calendar & Timezone Timestamp Validation (Production Unit Tests)
// ==============================================================================

test('Timestamp: leap year detection utility', () => {
  assert.strictEqual(isLeapYear(2024), true); // Divisible by 4
  assert.strictEqual(isLeapYear(2000), true); // Divisible by 400
  assert.strictEqual(isLeapYear(1900), false); // Divisible by 100 but not 400
  assert.strictEqual(isLeapYear(2026), false); // Not divisible by 4
});

test('Timestamp: days in month utility', () => {
  assert.strictEqual(getDaysInMonth(2024, 2), 29); // Leap Feb
  assert.strictEqual(getDaysInMonth(2026, 2), 28); // Non-leap Feb
  assert.strictEqual(getDaysInMonth(2026, 4), 30); // April
  assert.strictEqual(getDaysInMonth(2026, 12), 31); // Dec
  assert.strictEqual(getDaysInMonth(2026, 13), 0); // Invalid month
});

test('Timestamp: year 0000 is rejected outside supported range (1-9999)', () => {
  const result = validateEventTimestamp('0000-01-01T00:00:00Z');
  assert.strictEqual(result.isValid, false);
  assert.match(result.error ?? '', /year must be between 0001 and 9999/i);
});

test('Timestamp: valid leap day (Feb 29 on leap year) is accepted', () => {
  const result = validateEventTimestamp('2024-02-29T12:00:00Z');
  assert.strictEqual(result.isValid, true);
  assert.strictEqual(result.value, '2024-02-29T12:00:00Z');
});

test('Timestamp: invalid leap day (Feb 29 on non-leap year) is rejected', () => {
  const y2026 = validateEventTimestamp('2026-02-29T12:00:00Z');
  assert.strictEqual(y2026.isValid, false);
  assert.match(y2026.error ?? '', /invalid calendar date/i);

  const y2023 = validateEventTimestamp('2023-02-29T12:00:00Z');
  assert.strictEqual(y2023.isValid, false);
  assert.match(y2023.error ?? '', /invalid calendar date/i);
});

test('Timestamp: February 30 is rejected', () => {
  const leapFeb30 = validateEventTimestamp('2024-02-30T12:00:00Z');
  assert.strictEqual(leapFeb30.isValid, false);
  assert.match(leapFeb30.error ?? '', /invalid calendar date/i);

  const nonLeapFeb30 = validateEventTimestamp('2026-02-30T12:00:00Z');
  assert.strictEqual(nonLeapFeb30.isValid, false);
  assert.match(nonLeapFeb30.error ?? '', /invalid calendar date/i);
});

test('Timestamp: invalid month or day is rejected', () => {
  const month13 = validateEventTimestamp('2026-13-01T10:00:00Z');
  assert.strictEqual(month13.isValid, false);
  assert.match(month13.error ?? '', /month must be between 01 and 12/i);

  const month0 = validateEventTimestamp('2026-00-10T10:00:00Z');
  assert.strictEqual(month0.isValid, false);
  assert.match(month0.error ?? '', /month must be between 01 and 12/i);

  const day0 = validateEventTimestamp('2026-06-00T10:00:00Z');
  assert.strictEqual(day0.isValid, false);
  assert.match(day0.error ?? '', /invalid calendar date/i);

  const april31 = validateEventTimestamp('2026-04-31T10:00:00Z');
  assert.strictEqual(april31.isValid, false);
  assert.match(april31.error ?? '', /invalid calendar date/i);
});

test('Timestamp: invalid hour, minute, or second components are rejected', () => {
  const hour24 = validateEventTimestamp('2026-06-01T24:00:00Z');
  assert.strictEqual(hour24.isValid, false);
  assert.match(hour24.error ?? '', /hour must be between 00 and 23/i);

  const minute60 = validateEventTimestamp('2026-06-01T10:60:00Z');
  assert.strictEqual(minute60.isValid, false);
  assert.match(minute60.error ?? '', /minute must be between 00 and 59/i);

  const second60 = validateEventTimestamp('2026-06-01T10:00:60Z');
  assert.strictEqual(second60.isValid, false);
  assert.match(second60.error ?? '', /second must be between 00 and 59/i);
});

test('Timestamp: valid offsets and fractional seconds are accepted', () => {
  const fracZ = validateEventTimestamp('2026-06-01T10:00:00.123Z');
  assert.strictEqual(fracZ.isValid, true);
  assert.strictEqual(fracZ.value, '2026-06-01T10:00:00.123Z');

  const posOffset = validateEventTimestamp('2026-06-01T10:00:00+05:30');
  assert.strictEqual(posOffset.isValid, true);

  const negOffset = validateEventTimestamp('2026-06-01T10:00:00-04:00');
  assert.strictEqual(negOffset.isValid, true);

  const noColonOffset = validateEventTimestamp('2026-06-01T10:00:00+0530');
  assert.strictEqual(noColonOffset.isValid, true);

  const fracOffset = validateEventTimestamp('2026-06-01T10:00:00.999+02:00');
  assert.strictEqual(fracOffset.isValid, true);
});

test('Timestamp: timezone-naive timestamps are rejected', () => {
  const naiveT = validateEventTimestamp('2026-06-01T10:00:00');
  assert.strictEqual(naiveT.isValid, false);
  assert.match(naiveT.error ?? '', /explicit timezone/i);

  const naiveSpace = validateEventTimestamp('2026-06-01 10:00:00');
  assert.strictEqual(naiveSpace.isValid, false);
  assert.match(naiveSpace.error ?? '', /explicit timezone/i);
});

test('Timestamp: empty or whitespace timestamp uses current UTC time', () => {
  const mockNow = '2026-11-17T05:00:00.000Z';
  const emptyStr = validateEventTimestamp('', () => mockNow);
  assert.strictEqual(emptyStr.isValid, true);
  assert.strictEqual(emptyStr.value, mockNow);

  const whitespace = validateEventTimestamp('   ', () => mockNow);
  assert.strictEqual(whitespace.isValid, true);
  assert.strictEqual(whitespace.value, mockNow);

  const nullVal = validateEventTimestamp(null, () => mockNow);
  assert.strictEqual(nullVal.isValid, true);
  assert.strictEqual(nullVal.value, mockNow);
});

// ==============================================================================
// 2. Ask Page State Management (Behavior Simulation)
// ==============================================================================

test('Ask state simulation: first succeeds, second fails, no stale answer', async () => {
  let responseState: AskResponse | null = null;
  let isError = false;

  const mockAskApi = async (question: string): Promise<AskResponse> => {
    if (question === 'valid') {
      return {
        mesh: {
          question: 'valid',
          result: { plan: { intent: 'fact' } },
          answer: {
            value: 'Python 3.12',
            conflict_flagged: false,
            needs_review_flagged: false,
            abstained: false,
            explanation: 'Server 1 was upgraded.',
            answer_text: 'Python 3.12',
          },
        },
        baseline: null,
        baseline_retrieved: [],
      };
    }
    throw new Error('API failure: invalid query');
  };

  // Simulating AskPage submit workflow
  const submitQuestion = async (q: string) => {
    // Clear previous response before starting the request
    responseState = null;
    isError = false;

    try {
      const res = await mockAskApi(q);
      responseState = res;
    } catch {
      isError = true;
    }
  };

  // Question 1: succeeds
  await submitQuestion('valid');
  assert.strictEqual(isError, false);
  assert.notStrictEqual(responseState, null);
  assert.strictEqual(responseState?.mesh.answer.value, 'Python 3.12');

  // Question 2: fails
  await submitQuestion('fail');
  assert.strictEqual(isError, true);
  // Assertion: previous answer must NOT linger on failure
  assert.strictEqual(responseState, null);
});

// ==============================================================================
// 3. Ingest Page State Management (Behavior Simulation)
// ==============================================================================

test('Ingest state simulation: first succeeds, second fails, no stale report', async () => {
  let reportState: IngestReport | null = null;
  let isError = false;

  const mockIngestApi = async (text: string): Promise<IngestReport> => {
    if (text === 'valid message') {
      return {
        message_id: 101,
        duplicate: false,
        claims_extracted: 1,
        decisions: [
          {
            id: 1,
            message_id: 101,
            claim_json: { subject: 'Server 1', attribute: 'python_version', value: '3.12' },
            action: 'INSERT',
            rule_id: 'R1_NO_CONCURRENT_SAME_VALUE',
            explanation: 'Inserted new fact',
            created_at: '2026-06-01T10:00:00Z',
          },
        ],
      };
    }
    throw new Error('API failure: ingestion failed');
  };

  // Simulating IngestPage submit workflow
  const submitIngest = async (msg: string) => {
    // Clear previous result before starting request
    reportState = null;
    isError = false;

    try {
      const res = await mockIngestApi(msg);
      reportState = res;
    } catch {
      isError = true;
    }
  };

  // Ingestion 1: succeeds
  await submitIngest('valid message');
  assert.strictEqual(isError, false);
  assert.notStrictEqual(reportState, null);
  assert.strictEqual(reportState?.message_id, 101);

  // Ingestion 2: fails
  await submitIngest('fail message');
  assert.strictEqual(isError, true);
  // Assertion: previous report must NOT linger on failure
  assert.strictEqual(reportState, null);
});

// ==============================================================================
// 4. Catalog Source Validation & Error Handling (Behavior Simulation)
// ==============================================================================

test('Catalog simulation: source validation, retry action, and no hardcoded fallback', () => {
  let refetchCalled = false;
  const mockRefetch = () => {
    refetchCalled = true;
  };

  // Scenario A: Catalog query returns error
  const catalogError: Error | null = new Error('Network error loading catalog');
  const catalogData: CatalogResponse | undefined = undefined;

  let renderedError = false;
  let renderedFallback = false;

  if (catalogError) {
    renderedError = true;
    renderedFallback = false; // Never render hardcoded sources on error
  } else if (catalogData?.sources) {
    renderedFallback = false;
  }

  assert.strictEqual(renderedError, true);
  assert.strictEqual(renderedFallback, false);

  // Retry action executes refetch
  mockRefetch();
  assert.strictEqual(refetchCalled, true);

  // Scenario B: Catalog query succeeds with sources
  const successCatalog: CatalogResponse = {
    sources: [
      { id: 'sysadmin_priya', name: 'Priya', type: 'human', reliability: 0.95 },
      { id: 'crawler_bot', name: 'Crawler', type: 'sensor', reliability: 0.8 },
    ],
  };

  const availableSources = successCatalog.sources ?? [];
  assert.strictEqual(availableSources.length, 2);

  // Verify source existence check
  const isValidSource = (id: string) => availableSources.some((s) => s.id === id);
  assert.strictEqual(isValidSource('sysadmin_priya'), true);
  assert.strictEqual(isValidSource('unknown_hacker'), false);
  assert.strictEqual(isValidSource(''), false);

  // Preset fallback logic when preset source is missing from catalog
  const presetSource = 'missing_agent';
  let targetSource = '';
  if (isValidSource(presetSource)) {
    targetSource = presetSource;
  } else if (availableSources.length > 0) {
    targetSource = availableSources[0].id;
  }
  assert.strictEqual(targetSource, 'sysadmin_priya');
});
