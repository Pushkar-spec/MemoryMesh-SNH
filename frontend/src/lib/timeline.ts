import type { SlotTimeline } from '../api/client';

export interface TimelineLaneStats {
  activeCount: number;
  historicalCount: number;
  disputedCount: number;
  rejectedCount: number;
  totalCount: number;
}

/**
 * Generate a unique, deterministic composite key for a slot timeline lane.
 * Serializes [entity, attribute, context_key] to avoid collision across delimiters.
 */
export function getTimelineLaneKey(lane: SlotTimeline): string {
  return JSON.stringify([lane.entity, lane.attribute, lane.context_key || '']);
}

/**
 * Compute summary statistics for a slot timeline lane.
 * Strictly separates active, historical, disputed, and rejected memories.
 */
export function getTimelineLaneStats(lane: SlotTimeline): TimelineLaneStats {
  const memories = lane.memories ?? [];
  const rejected = lane.rejected ?? [];

  const activeCount = memories.filter(
    (m) => m.memory.status === 'active'
  ).length;
  const historicalCount = memories.filter(
    (m) => m.memory.status === 'historical'
  ).length;
  const disputedCount = memories.filter(
    (m) => m.memory.status === 'disputed'
  ).length;
  const rejectedCount = rejected.length;

  return {
    activeCount,
    historicalCount,
    disputedCount,
    rejectedCount,
    totalCount: memories.length + rejectedCount,
  };
}
