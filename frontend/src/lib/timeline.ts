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
 * Strictly separates active, historical, disputed, and rejected memories:
 * - activeCount: count of memories with status === 'active'
 * - historicalCount: count of memories with status === 'historical'
 * - disputedCount: count of memories with status === 'disputed'
 * - rejectedCount: count of memories in lane.rejected
 * - totalCount: total count of all memories in the lane (memories + rejected)
 */
export function getTimelineLaneStats(lane: SlotTimeline): TimelineLaneStats {
  const memories = lane.memories ?? [];
  const rejected = lane.rejected ?? [];

  const activeCount = memories.filter(
    (m) => m.memory?.status === 'active'
  ).length;
  const historicalCount = memories.filter(
    (m) => m.memory?.status === 'historical'
  ).length;
  const disputedCount = memories.filter(
    (m) => m.memory?.status === 'disputed'
  ).length;

  // Track unique memories to ensure totalCount does not double-count
  const seenMemoryIds = new Set<number>();
  let unkeyedCount = 0;

  for (const item of memories) {
    if (item.memory?.id != null) {
      seenMemoryIds.add(item.memory.id);
    } else {
      unkeyedCount++;
    }
  }

  // Count rejected items: lane.rejected plus any rejected status in memories,
  // avoiding double-counting if an item is present in both collections
  const rejectedIds = new Set<number>();
  let unkeyedRejected = 0;

  for (const item of rejected) {
    if (item.memory?.id != null) {
      rejectedIds.add(item.memory.id);
      seenMemoryIds.add(item.memory.id);
    } else {
      unkeyedRejected++;
    }
  }

  for (const item of memories) {
    if (item.memory?.status === 'rejected') {
      if (item.memory.id != null) {
        rejectedIds.add(item.memory.id);
      } else {
        unkeyedRejected++;
      }
    }
  }

  const rejectedCount = rejectedIds.size + unkeyedRejected;
  const totalCount = seenMemoryIds.size + unkeyedCount + unkeyedRejected;

  return {
    activeCount,
    historicalCount,
    disputedCount,
    rejectedCount,
    totalCount,
  };
}
