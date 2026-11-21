/**
 * Client-side Server-Sent Events (SSE) scenario replay utility for MemoryMesh.
 *
 * Governed by docs/SPECIFICATION.md Section 7.2 and docs/ARCHITECTURE.md.
 */

import type { IngestReport } from '../api/client';

export interface ReplayOptions {
  scenario?: string;
  mode?: 'structured' | 'e2e';
  delay_ms?: number;
  /** @deprecated Replay is read-only; reset is performed via POST /api/demo/reset. */
  reset?: boolean;
  baseUrl?: string;
  onMessageIngested?: (report: IngestReport) => void;
  onDone?: (data: { messages: number }) => void;
  onError?: (err: { error: string } | Error) => void;
  onOpen?: () => void;
}

export interface ReplaySession {
  stop: () => void;
  readonly isActive: boolean;
}

/**
 * Initiates an SSE connection to `/api/demo/replay` and streams ingestion reports.
 *
 * @param options Replay parameters, event callbacks, and network configuration.
 * @returns ReplaySession handle providing active status check and abort capability.
 */
export function startReplay(options: ReplayOptions = {}): ReplaySession {
  const {
    scenario = 'hero-001',
    mode = 'structured',
    delay_ms = 800,
    baseUrl = '',
    onMessageIngested,
    onDone,
    onError,
    onOpen,
  } = options;

  const params = new URLSearchParams({
    scenario,
    mode,
    delay_ms: String(delay_ms),
  });

  const url = `${baseUrl}/api/demo/replay?${params.toString()}`;
  let isActive = true;
  let eventSource: EventSource | null = null;

  try {
    eventSource = new EventSource(url);
  } catch (err) {
    isActive = false;
    onError?.(err instanceof Error ? err : new Error(String(err)));
    return {
      stop: () => {},
      get isActive() {
        return false;
      },
    };
  }

  const stop = () => {
    if (isActive) {
      isActive = false;
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
    }
  };

  eventSource.onopen = () => {
    if (isActive) {
      onOpen?.();
    }
  };

  eventSource.addEventListener('message_ingested', (event: MessageEvent) => {
    if (!isActive) return;
    try {
      const report = JSON.parse(event.data) as IngestReport;
      onMessageIngested?.(report);
    } catch (parseErr) {
      onError?.(parseErr instanceof Error ? parseErr : new Error('Malformed report JSON.'));
    }
  });

  eventSource.addEventListener('done', (event: MessageEvent) => {
    if (!isActive) return;
    try {
      const data = JSON.parse(event.data) as { messages: number };
      stop();
      onDone?.(data);
    } catch (parseErr) {
      stop();
      onError?.(parseErr instanceof Error ? parseErr : new Error('Malformed done event JSON.'));
    }
  });

  eventSource.addEventListener('error', (event: Event) => {
    if (!isActive) return;
    const messageEvent = event as MessageEvent;
    if (messageEvent.data) {
      try {
        const errPayload = JSON.parse(messageEvent.data) as { error: string };
        stop();
        onError?.(errPayload);
        return;
      } catch {
        // Fall back to generic connection failure
      }
    }
    stop();
    onError?.(new Error('EventSource encountered an error or connection was closed.'));
  });

  return {
    stop,
    get isActive() {
      return isActive;
    },
  };
}
