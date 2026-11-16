/**
 * Utility functions for date, number, confidence, and latency formatting.
 *
 * Governed by docs/SPECIFICATION.md Section 7.5.
 */

/**
 * Format an ISO UTC date string into a readable UTC timestamp.
 * Example: "2026-06-01T12:00:00Z" -> "2026-06-01 12:00:00 UTC"
 */
export function formatUtcDateTime(dateStr: string | null | undefined): string {
  if (!dateStr) return '—';
  try {
    const dt = new Date(dateStr);
    if (isNaN(dt.getTime())) return dateStr;
    const pad = (n: number) => n.toString().padStart(2, '0');
    const yyyy = dt.getUTCFullYear();
    const mm = pad(dt.getUTCMonth() + 1);
    const dd = pad(dt.getUTCDate());
    const hh = pad(dt.getUTCHours());
    const min = pad(dt.getUTCMinutes());
    const ss = pad(dt.getUTCSeconds());
    return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss} UTC`;
  } catch {
    return dateStr;
  }
}

/**
 * Format an ISO date string into a short date: "2026-06-01"
 */
export function formatShortDate(dateStr: string | null | undefined): string {
  if (!dateStr) return '—';
  try {
    const dt = new Date(dateStr);
    if (isNaN(dt.getTime())) return dateStr;
    return dt.toISOString().split('T')[0];
  } catch {
    return dateStr;
  }
}

/**
 * Format confidence score (0.0 to 1.0) as percentage string.
 * Example: 0.954 -> "95.4%"
 */
export function formatConfidence(score: number | null | undefined, decimals = 1): string {
  if (score === null || score === undefined || isNaN(score)) return '—';
  const pct = score * 100;
  return `${pct.toFixed(decimals)}%`;
}

/**
 * Format execution latency in milliseconds.
 * Example: 42.5 -> "43ms", 1250 -> "1.25s"
 */
export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Format a validity interval: [valid_from, valid_to]
 * Example: [2026-06-01, null] -> "2026-06-01 → Present"
 */
export function formatValidityInterval(
  from: string | null | undefined,
  to: string | null | undefined
): string {
  const fromStr = formatShortDate(from);
  const toStr = to ? formatShortDate(to) : 'Present';
  return `${fromStr} → ${toStr}`;
}
