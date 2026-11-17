/**
 * TanStack Query hooks and query-key infrastructure for MemoryMesh.
 *
 * Governed by docs/ARCHITECTURE.md Section 4b, docs/ROADMAP_AND_TASKS.md T9.2/T9.3,
 * and canonical plan P9.2/P9.3.
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
  type MemoryView,
  type QueryResult,
  type IngestRequest,
  type IngestReport,
  type AskRequest,
  type AskResponse,
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
// Shell Query Hooks (P9.2)
// ==============================================================================

/**
 * Fetch backend health, LLM provider info, and database status.
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
// Memories & Provenance Query Hooks (P9.3)
// ==============================================================================

/**
 * List memories with optional filtering by entity, attribute, status, needs_review.
 */
export function useMemories(
  params?: {
    entity?: string;
    attribute?: string;
    status?: string;
    needs_review?: boolean;
    now?: string;
    limit?: number;
    offset?: number;
  },
  options?: Omit<UseQueryOptions<MemoryView[], ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<MemoryView[], ApiError>({
    queryKey: queryKeys.memories(params),
    queryFn: () => api.getMemories(params),
    ...options,
  });
}

/**
 * Retrieve a single memory by ID.
 */
export function useMemory(
  id: number,
  now?: string,
  options?: Omit<UseQueryOptions<MemoryView, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<MemoryView, ApiError>({
    queryKey: queryKeys.memory(id, now),
    queryFn: () => api.getMemory(id, now),
    enabled: id > 0,
    ...options,
  });
}

/**
 * Explain provenance and decision reasoning for a memory.
 */
export function useMemoryExplain(
  id: number,
  options?: Omit<UseQueryOptions<QueryResult, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<QueryResult, ApiError>({
    queryKey: queryKeys.memoryExplain(id),
    queryFn: () => api.explainMemory(id),
    enabled: id > 0,
    ...options,
  });
}

// ==============================================================================
// Mutation Hooks (P9.2 / P9.3)
// ==============================================================================

/**
 * Reset database and re-seed demo data.
 * Invalidates all queries across the application.
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

/**
 * Ingest message and extract/resolve memory claims (P9.3).
 * On success, invalidates memory and decision queries.
 */
export function useIngest(
  options?: UseMutationOptions<IngestReport, ApiError, IngestRequest>
) {
  const queryClient = useQueryClient();
  return useMutation<IngestReport, ApiError, IngestRequest>({
    mutationFn: (body) => api.ingest(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memories'] });
      queryClient.invalidateQueries({ queryKey: ['decisions'] });
    },
    ...options,
  });
}

/**
 * Ask a natural language question with optional naive RAG comparison (P9.3).
 */
export function useAsk(
  options?: UseMutationOptions<AskResponse, ApiError, AskRequest>
) {
  return useMutation<AskResponse, ApiError, AskRequest>({
    mutationFn: (body) => api.ask(body),
    ...options,
  });
}
