import type { DisputeView, MemoryView, DecisionLogEntry } from '../api/client';

export interface ReviewQueueCounts {
  disputesCount: number;
  flaggedCount: number;
  totalPending: number;
}

/**
 * Compute aggregate counts for pending review items.
 */
export function computeReviewQueueCounts(
  disputes: DisputeView[] = [],
  flaggedMemories: MemoryView[] = []
): ReviewQueueCounts {
  const disputesCount = (disputes ?? []).length;
  const flaggedCount = (flaggedMemories ?? []).length;
  return {
    disputesCount,
    flaggedCount,
    totalPending: disputesCount + flaggedCount,
  };
}

/**
 * Filter decision records by rule ID (case-insensitive).
 * Empty or undefined ruleId returns all decisions unfiltered.
 */
export function filterDecisionsByRule(
  decisions: DecisionLogEntry[] = [],
  ruleId?: string
): DecisionLogEntry[] {
  if (!ruleId || ruleId.trim() === '') {
    return decisions ?? [];
  }
  const target = ruleId.trim().toUpperCase();
  return (decisions ?? []).filter(
    (d) => (d.rule_id || '').toUpperCase() === target
  );
}

/**
 * Extract sorted unique set of available rule IDs from decisions and canonical rules.
 */
export function extractAvailableRules(
  decisions: DecisionLogEntry[] = [],
  canonicalRules: readonly string[] = []
): string[] {
  const rules = new Set<string>(canonicalRules);
  for (const d of decisions ?? []) {
    if (d.rule_id) {
      rules.add(d.rule_id);
    }
  }
  return Array.from(rules).sort();
}
