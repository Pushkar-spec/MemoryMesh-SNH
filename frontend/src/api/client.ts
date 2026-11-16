/**
 * Centralized API client for MemoryMesh FastAPI backend.
 *
 * Governed by docs/ARCHITECTURE.md Section 4b and docs/SPECIFICATION.md Section 7.
 */

import type { components } from './types';

export type Schemas = components['schemas'];
export type HealthResponse = Schemas['HealthResponse'];
export type CatalogResponse = Schemas['CatalogResponse'];
export type Memory = Schemas['Memory'];
export type MemoryView = Schemas['MemoryView'];
export type DecisionLogEntry = Schemas['DecisionLogEntry'];
export type DisputeView = Schemas['DisputeView'];
export type GraphResponse = Schemas['GraphResponse'];
export type SlotTimeline = Schemas['SlotTimeline'];
export type IngestRequest = Schemas['IngestRequest'];
export type IngestReport = Schemas['IngestReport'];
export type AskRequest = Schemas['AskRequest'];
export type AskResponse = Schemas['AskResponse'];
export type LLMCallLog = Schemas['LLMCallLog'];
export type EvalSummary = Schemas['EvalSummary'];
export type ResetResponse = Schemas['ResetResponse'];
export type ResolveDisputeRequest = Schemas['ResolveDisputeRequest'];
export type ImpactItem = Schemas['ImpactItem'];
export type QueryResult = Schemas['QueryResult'];

/**
 * Standardized API Error matching backend error envelope.
 * Backend envelope format: {"error": {"code": "string", "message": "string"}}
 */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: unknown;

  constructor(message: string, code = 'unknown_error', status = 500, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

/**
 * Resolve API base URL.
 * Defaults to empty string to use Vite's dev proxy (/api -> http://localhost:8000),
 * or respects VITE_API_BASE when provided in environment.
 */
function getApiBaseUrl(): string {
  const envBase = import.meta.env.VITE_API_BASE;
  if (!envBase) return '';
  // Strip trailing slash if present
  return envBase.endsWith('/') ? envBase.slice(0, -1) : envBase;
}

/**
 * Centralized fetch wrapper handling URL resolution, JSON serialization,
 * HTTP status validation, and error mapping.
 */
export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const base = getApiBaseUrl();
  const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${base}${normalizedEndpoint}`;

  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (options.body && typeof options.body === 'string' && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch (networkErr) {
    const errorMsg =
      networkErr instanceof Error ? networkErr.message : 'Network connection failed';
    throw new ApiError(
      `Unable to connect to MemoryMesh backend: ${errorMsg}`,
      'network_error',
      0
    );
  }

  // Handle successful text/json responses
  if (response.ok) {
    if (response.status === 204 || response.headers.get('content-length') === '0') {
      return {} as T;
    }
    const text = await response.text();
    if (!text) {
      return {} as T;
    }
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      try {
        return JSON.parse(text) as T;
      } catch {
        return text as unknown as T;
      }
    }
    return text as unknown as T;
  }

  // Parse error envelope
  let errorData: unknown = null;
  try {
    errorData = await response.json();
  } catch {
    errorData = null;
  }

  let errorCode = 'http_error';
  let errorMessage = `HTTP ${response.status}: ${response.statusText || 'Request failed'}`;

  if (
    errorData &&
    typeof errorData === 'object' &&
    'error' in errorData &&
    typeof (errorData as { error: unknown }).error === 'object'
  ) {
    const envelope = (errorData as { error: { code?: string; message?: string } }).error;
    if (envelope.code) errorCode = envelope.code;
    if (envelope.message) errorMessage = envelope.message;
  } else if (
    errorData &&
    typeof errorData === 'object' &&
    'detail' in errorData
  ) {
    const detail = (errorData as { detail: unknown }).detail;
    errorCode = response.status === 422 ? 'validation_error' : 'http_error';
    if (typeof detail === 'string') {
      errorMessage = detail;
    } else if (Array.isArray(detail)) {
      errorMessage = detail
        .map((d: { loc?: unknown[]; msg?: string }) => {
          const field = Array.isArray(d.loc)
            ? d.loc.filter((part) => part !== 'body').join('.')
            : '';
          return field ? `${field}: ${d.msg}` : d.msg || 'Invalid field';
        })
        .join('; ');
    } else {
      errorMessage = JSON.stringify(detail);
    }
  } else if (response.status === 500) {
    errorCode = 'internal';
    errorMessage = 'An internal server error occurred while processing the request.';
  }

  throw new ApiError(errorMessage, errorCode, response.status, errorData);
}

/**
 * Query parameter serializer helper.
 */
function toQueryString(params?: Record<string, string | number | boolean | null | undefined>): string {
  if (!params) return '';
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') {
      searchParams.append(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return qs ? `?${qs}` : '';
}

/**
 * Centralized API service object with strongly typed endpoint calls.
 */
export const api = {
  // Read operations
  getHealth: (): Promise<HealthResponse> =>
    apiRequest<HealthResponse>('/api/health'),

  getCatalog: (): Promise<CatalogResponse> =>
    apiRequest<CatalogResponse>('/api/catalog'),

  getMemories: (params?: {
    entity?: string;
    attribute?: string;
    status?: string;
    needs_review?: boolean;
    now?: string;
    limit?: number;
    offset?: number;
  }): Promise<MemoryView[]> =>
    apiRequest<MemoryView[]>(`/api/memories${toQueryString(params)}`),

  getMemory: (id: number, now?: string): Promise<MemoryView> =>
    apiRequest<MemoryView>(`/api/memories/${id}${toQueryString({ now })}`),

  explainMemory: (id: number): Promise<QueryResult> =>
    apiRequest<QueryResult>(`/api/memories/${id}/explain`),

  getMemoryImpact: (id: number): Promise<ImpactItem[]> =>
    apiRequest<ImpactItem[]>(`/api/memories/${id}/impact`),

  getTimeline: (params?: { entity?: string; attribute?: string }): Promise<SlotTimeline> =>
    apiRequest<SlotTimeline>(`/api/timeline${toQueryString(params)}`),

  getGraph: (params?: { seed_id?: number; depth?: number }): Promise<GraphResponse> =>
    apiRequest<GraphResponse>(`/api/graph${toQueryString(params)}`),

  getDecisions: (params?: {
    action?: string;
    rule_id?: string;
    entity?: string;
    limit?: number;
    offset?: number;
  }): Promise<DecisionLogEntry[]> =>
    apiRequest<DecisionLogEntry[]>(`/api/decisions${toQueryString(params)}`),

  getLLMCalls: (params?: {
    prompt_id?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<LLMCallLog[]> =>
    apiRequest<LLMCallLog[]>(`/api/llm/calls${toQueryString(params)}`),

  getEvalSummary: (): Promise<EvalSummary> =>
    apiRequest<EvalSummary>('/api/eval/latest'),

  getEvalReport: (): Promise<string> =>
    apiRequest<string>('/api/eval/report'),

  // Write operations
  ingest: (body: IngestRequest): Promise<IngestReport> =>
    apiRequest<IngestReport>('/api/ingest', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  acknowledgeMemory: (
    id: number
  ): Promise<{ success: boolean; memory_id: number; unflagged: boolean }> =>
    apiRequest<{ success: boolean; memory_id: number; unflagged: boolean }>(
      `/api/memories/${id}/acknowledge`,
      { method: 'POST' }
    ),

  resolveDispute: (
    body: ResolveDisputeRequest
  ): Promise<{
    success: boolean;
    dispute_id: number;
    winner_memory_id: number;
    affected_memory_ids: number[];
  }> =>
    apiRequest<{
      success: boolean;
      dispute_id: number;
      winner_memory_id: number;
      affected_memory_ids: number[];
    }>('/api/disputes/resolve', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  resetDemo: (): Promise<ResetResponse> =>
    apiRequest<ResetResponse>('/api/demo/reset', {
      method: 'POST',
    }),

  ask: (body: AskRequest): Promise<AskResponse> =>
    apiRequest<AskResponse>('/api/ask', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
};
