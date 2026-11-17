/**
 * Validation utilities for Ingest and Query forms.
 * Governed by docs/SPECIFICATION.md Section 7.
 */

export interface TimestampValidationResult {
  isValid: boolean;
  value?: string;
  error?: string;
}

/**
 * Check if a given year is a leap year.
 */
export function isLeapYear(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

/**
 * Return maximum calendar days for a given year and month (1-12).
 */
export function getDaysInMonth(year: number, month: number): number {
  switch (month) {
    case 1:
    case 3:
    case 5:
    case 7:
    case 8:
    case 10:
    case 12:
      return 31;
    case 4:
    case 6:
    case 9:
    case 11:
      return 30;
    case 2:
      return isLeapYear(year) ? 29 : 28;
    default:
      return 0;
  }
}

/**
 * Validate an event timestamp string.
 *
 * Rules:
 * - Empty or whitespace-only input returns current UTC ISO string.
 * - Timezone-aware timestamp with explicit 'Z' or valid offset (+/-HH:MM or +/-HHMM) is accepted.
 * - Timezone-naive timestamp (e.g. 2026-06-01T10:00:00) is rejected.
 * - Fractional seconds (.sss) are supported.
 * - Impossible calendar dates (Feb 30, invalid leap days, invalid month/day) are rejected.
 * - Invalid time components (hours 0-23, minutes 0-59, seconds 0-59) are rejected.
 */
export function validateEventTimestamp(
  rawInput: string | null | undefined,
  nowFn: () => string = () => new Date().toISOString()
): TimestampValidationResult {
  const trimmed = rawInput?.trim() ?? '';
  if (!trimmed) {
    return { isValid: true, value: nowFn() };
  }

  // Regex enforcing ISO structure and explicit timezone (Z or +/-HH:MM or +/-HHMM)
  const ISO_TIMESTAMP_REGEX = new RegExp(
    '^(\\d{4})-(\\d{2})-(\\d{2})[T ](\\d{2}):(\\d{2})' +
      '(?::(\\d{2})(?:\\.\\d+)?)?' +
      '(Z|[+-](?:[01]\\d|2[0-3]):?[0-5]\\d)$',
    'i',
  );

  const match = ISO_TIMESTAMP_REGEX.exec(trimmed);
  if (!match) {
    return {
      isValid: false,
      error:
        'Timestamp must include an explicit timezone (e.g. 2026-06-01T10:00:00Z or +00:00).',
    };
  }

  const year = parseInt(match[1], 10);
  const month = parseInt(match[2], 10);
  const day = parseInt(match[3], 10);
  const hour = parseInt(match[4], 10);
  const minute = parseInt(match[5], 10);
  const second = match[6] !== undefined ? parseInt(match[6], 10) : 0;

  if (year < 1 || year > 9999) {
    return {
      isValid: false,
      error: 'Timestamp year must be between 0001 and 9999.',
    };
  }

  if (month < 1 || month > 12) {
    return {
      isValid: false,
      error: 'Timestamp month must be between 01 and 12.',
    };
  }

  const maxDays = getDaysInMonth(year, month);
  if (day < 1 || day > maxDays) {
    return {
      isValid: false,
      error: `Invalid calendar date: ${year}-${String(month).padStart(2, '0')}-${String(
        day
      ).padStart(2, '0')} does not exist.`,
    };
  }

  if (hour < 0 || hour > 23) {
    return {
      isValid: false,
      error: 'Timestamp hour must be between 00 and 23.',
    };
  }

  if (minute < 0 || minute > 59) {
    return {
      isValid: false,
      error: 'Timestamp minute must be between 00 and 59.',
    };
  }

  if (second < 0 || second > 59) {
    return {
      isValid: false,
      error: 'Timestamp second must be between 00 and 59.',
    };
  }

  if (isNaN(Date.parse(trimmed))) {
    return {
      isValid: false,
      error: 'Timestamp is not a valid date.',
    };
  }

  return { isValid: true, value: trimmed };
}
