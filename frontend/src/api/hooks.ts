/**
 * Foundational TanStack Query hooks and query-key infrastructure for MemoryMesh.
 *
 * Governed by docs/ARCHITECTURE.md Section 4b, docs/ROADMAP_AND_TASKS.md T9.2,
 * and canonical plan P9.2.
 *
 * Feature-specific hooks are deferred to their respective implementation batches:
 * - P9.3: useIngest, useMemories, useMemory, useAsk
 * - P9.4: useGraph, useTimeline
 * - P9.5: useResolveDispute, useAcknowledgeMemory, useDecisions
 * - P9.7: useLLMCalls, useEvalSummary, useEvalReport
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
  type UseQueryOptions,
  type UseMutationOptions,
} from '@tanstack/react-query';
import {
  api,
  type HealthResponse,
  type CatalogResponse,
  type ResetResponse,
  type ApiError,
} from './client';

/**
 * Standardized query key factory establishing conventions for caching and invalidation.
 */
export const queryKeys = {
  health: ['health'] as const,
  catalog: ['catalog'] as const,
  memories: (params?: Record<string, unknown>) => ['memories', params] as const,
  memory: (id: number, now?: string) => ['memory', id, now] as const,
  memoryExplain: (id: number) => ['memory', id, 'explain'] as const,
  memoryImpact: (id: number) => ['memory', id, 'impact'] as const,
  timeline: (params?: { entity?: string; attribute?: string }) =>
    ['timeline', params] as const,
  graph: (params?: { seed_id?: number; depth?: number }) =>
    ['graph', params] as const,
  decisions: (params?: Record<string, unknown>) => ['decisions', params] as const,
  llmCalls: (params?: Record<string, unknown>) => ['llmCalls', params] as const,
  evalSummary: ['evalSummary'] as const,
  evalReport: ['evalReport'] as const,
};

// ==============================================================================
// Foundational Shell Query Hooks (P9.2)
// ==============================================================================

/**
 * Fetch backend health, LLM provider info, and database status.
 * Used by Header provider badge and Overview status card.
 */
export function useHealth(
  options?: Omit<UseQueryOptions<HealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HealthResponse, ApiError>({
    queryKey: queryKeys.health,
    queryFn: () => api.getHealth(),
    refetchInterval: 30000,
    ...options,
  });
}

/**
 * Fetch configuration catalog (sources, attributes, entities, thresholds).
 * Used by application overview and shell configuration inspectors.
 */
export function useCatalog(
  options?: Omit<UseQueryOptions<CatalogResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<CatalogResponse, ApiError>({
    queryKey: queryKeys.catalog,
    queryFn: () => api.getCatalog(),
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}

// ==============================================================================
// Foundational Shell Mutation Hooks (P9.2)
// ==============================================================================

/**
 * Reset database and re-seed demo data.
 * Explicitly required by App Shell Header per T9.2 ("header (demo clock, provider badge, Reset demo)").
 * On success, invalidates all queries across the application.
 */
export function useResetDemo(
  options?: UseMutationOptions<ResetResponse, ApiError, void>
) {
  const queryClient = useQueryClient();
  return useMutation<ResetResponse, ApiError, void>({
    mutationFn: () => api.resetDemo(),
    onSuccess: () => {
      queryClient.invalidateQueries();
    },
    ...options,
  });
}
