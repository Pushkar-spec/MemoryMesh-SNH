import React from 'react';
import {
  X,
  Database,
  FileText,
  Activity,
  Calendar,
  Layers,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';
import { useMemory, useMemoryExplain } from '../../api/hooks';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { ErrorState } from '../common/ErrorState';
import { StatusBadge } from '../common/StatusBadge';
import { ConfidenceBar } from '../common/ConfidenceBar';
import { ValidityBar } from '../common/ValidityBar';
import { DecisionCard } from '../common/DecisionCard';
import { formatUtcDateTime } from '../../lib/format';

interface ProvenanceDrawerProps {
  memoryId: number | null;
  onClose: () => void;
  onSelectMemory?: (id: number) => void;
}

export function ProvenanceDrawer({
  memoryId,
  onClose,
  onSelectMemory,
}: ProvenanceDrawerProps): React.JSX.Element | null {
  if (!memoryId) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over panel */}
      <div
        className="relative w-full max-w-2xl bg-slate-900 border-l border-slate-800
                   shadow-2xl flex flex-col h-full z-10 overflow-hidden"
      >
        <ProvenanceContent
          key={memoryId}
          memoryId={memoryId}
          onClose={onClose}
          onSelectMemory={onSelectMemory}
        />
      </div>
    </div>
  );
}

function ProvenanceContent({
  memoryId,
  onClose,
  onSelectMemory,
}: {
  memoryId: number;
  onClose: () => void;
  onSelectMemory?: (id: number) => void;
}): React.JSX.Element {
  const {
    data: memoryView,
    isLoading: memoryLoading,
    error: memoryError,
    refetch: refetchMemory,
  } = useMemory(memoryId);

  const {
    data: explainResult,
    isLoading: explainLoading,
    error: explainError,
    refetch: refetchExplain,
  } = useMemoryExplain(memoryId);

  const mem = memoryView?.memory;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div
        className="px-6 py-4 border-b border-slate-800 flex items-center
                   justify-between bg-slate-950/50 shrink-0"
      >
        <div className="flex items-center space-x-2.5">
          <div
            className="p-1.5 bg-indigo-600/30 border border-indigo-500/40
                       rounded-lg text-indigo-400"
          >
            <Database className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-base font-bold text-white tracking-tight">
                Memory #{memoryId} Provenance
              </h3>
              {mem?.status && (
                <StatusBadge status={mem.status} needsReview={mem.needs_review} />
              )}
            </div>
            <p className="text-xs text-slate-400">
              Audit trail, supporting evidence, and resolution decisions
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="p-1.5 text-slate-400 hover:text-white rounded-lg
                     hover:bg-slate-800 transition"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Body Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {memoryLoading ? (
          <div className="py-12 flex justify-center">
            <LoadingSpinner size="lg" label="Loading memory details..." />
          </div>
        ) : memoryError ? (
          <ErrorState
            error={memoryError}
            title="Failed to load memory details"
            onRetry={() => refetchMemory()}
          />
        ) : mem ? (
          <>
            {/* Needs Review Alert Banner */}
            {mem.needs_review && (
              <div
                className="p-3.5 rounded-xl border border-rose-800/80 bg-rose-950/40
                           text-rose-200 text-xs flex items-start space-x-2.5"
              >
                <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-rose-300">
                    Flagged for Human Review
                  </span>
                  {mem.review_reason && (
                    <p className="text-rose-200/90 mt-0.5 font-mono">
                      {mem.review_reason}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Core Memory Identity Grid */}
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 space-y-3">
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-slate-400">Entity:</span>
                  <p className="text-sm font-semibold text-white mt-0.5">
                    {mem.entity_name || `Entity #${mem.entity_id}`}
                  </p>
                </div>
                <div>
                  <span className="text-slate-400">Attribute:</span>
                  <p className="text-sm font-mono font-semibold text-indigo-300 mt-0.5">
                    {mem.attribute}
                  </p>
                </div>
                <div>
                  <span className="text-slate-400">Resolved Value:</span>
                  <p className="text-sm font-mono font-bold text-emerald-400 mt-0.5">
                    {mem.value_display || mem.value_norm}
                  </p>
                </div>
                <div>
                  <span className="text-slate-400">Confidence:</span>
                  <div className="mt-1">
                    <ConfidenceBar
                      score={memoryView?.effective_confidence ?? mem.confidence}
                    />
                  </div>
                </div>
              </div>

              {/* Context metadata */}
              {mem.context && Object.keys(mem.context).length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 text-xs">
                  <span className="text-slate-400">Context:</span>
                  <div className="flex flex-wrap gap-1.5 mt-1 font-mono">
                    {Object.entries(mem.context).map(([k, v]) => (
                      <span
                        key={k}
                        className="px-2 py-0.5 rounded bg-slate-900 border
                                   border-slate-800 text-slate-300"
                      >
                        {k}={String(v)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Validity and recording timeline */}
              <div
                className="pt-2 border-t border-slate-800/80 grid grid-cols-1
                           sm:grid-cols-2 gap-3 text-xs"
              >
                <div>
                  <span className="text-slate-400">Validity Interval:</span>
                  <div className="mt-1">
                    <ValidityBar validFrom={mem.valid_from} validTo={mem.valid_to} />
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Recorded At:</span>
                  <p className="text-xs font-mono text-slate-300 mt-1 flex items-center space-x-1">
                    <Calendar className="w-3.5 h-3.5 text-slate-400" />
                    <span>{formatUtcDateTime(mem.recorded_at)}</span>
                  </p>
                </div>
              </div>
            </div>

            {/* Supporting Evidence Records */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4
                  className="text-xs font-bold uppercase tracking-wider text-slate-400
                             flex items-center space-x-1.5"
                >
                  <FileText className="w-4 h-4 text-indigo-400" />
                  <span>Persisted Evidence ({mem.evidence?.length ?? 0})</span>
                </h4>
                <span className="text-[11px] font-mono text-slate-500">Source Citations</span>
              </div>

              {mem.evidence && mem.evidence.length > 0 ? (
                <div className="space-y-2">
                  {mem.evidence.map((ev) => (
                    <div
                      key={ev.id}
                      className="p-3 rounded-lg border border-slate-800 bg-slate-950/40
                                 text-xs space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className="px-2 py-0.5 rounded bg-slate-800 font-semibold text-slate-200"
                        >
                          {ev.source_id}
                        </span>
                        <span className="font-mono text-slate-400 text-[11px]">
                          Msg #{ev.message_id} &bull; {formatUtcDateTime(ev.observed_at)}
                        </span>
                      </div>
                      {ev.span && (
                        <p
                          className="text-xs text-slate-300 italic font-serif bg-slate-900/60
                                     p-2 rounded border border-slate-800"
                        >
                          &ldquo;{ev.span}&rdquo;
                        </p>
                      )}
                      <div
                        className="flex items-center justify-between text-[11px]
                                   text-slate-400 font-mono"
                      >
                        <span>Evidence Strength:</span>
                        <span className="text-slate-200 font-semibold">
                          {(ev.strength * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div
                  className="p-4 rounded-lg border border-dashed border-slate-800
                             text-center text-xs text-slate-500"
                >
                  No persisted raw evidence spans attached to this memory.
                </div>
              )}
            </div>

            {/* Decision History & Rule firings */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4
                  className="text-xs font-bold uppercase tracking-wider text-slate-400
                             flex items-center space-x-1.5"
                >
                  <Activity className="w-4 h-4 text-indigo-400" />
                  <span>Resolution Decisions ({explainResult?.decisions?.length ?? 0})</span>
                </h4>
                <span className="text-[11px] font-mono text-slate-500">Audit History</span>
              </div>

              {explainLoading ? (
                <LoadingSpinner size="sm" label="Loading decision history..." />
              ) : explainError ? (
                <ErrorState
                  error={explainError}
                  title="Failed to load decision history"
                  onRetry={() => refetchExplain()}
                />
              ) : explainResult?.decisions && explainResult.decisions.length > 0 ? (
                <div className="space-y-2">
                  {explainResult.decisions.map((dec) => (
                    <DecisionCard
                      key={dec.id}
                      decision={dec}
                      onMemoryClick={onSelectMemory}
                    />
                  ))}
                </div>
              ) : (
                <div
                  className="p-4 rounded-lg border border-dashed border-slate-800
                             text-center text-xs text-slate-500"
                >
                  No decision log entries linked to this memory.
                </div>
              )}
            </div>

            {/* Related Graph Edges */}
            {explainResult?.edges && explainResult.edges.length > 0 && (
              <div className="space-y-3">
                <h4
                  className="text-xs font-bold uppercase tracking-wider text-slate-400
                             flex items-center space-x-1.5"
                >
                  <Layers className="w-4 h-4 text-indigo-400" />
                  <span>Graph Relationships ({explainResult.edges.length})</span>
                </h4>
                <div className="space-y-1.5">
                  {explainResult.edges.map((edge) => (
                    <div
                      key={edge.id}
                      className="p-2.5 rounded-lg border border-slate-800 bg-slate-950/40
                                 text-xs flex items-center justify-between font-mono"
                    >
                      <div className="flex items-center space-x-2">
                        <span className="text-indigo-400">
                          mem #{edge.src_memory_id}
                        </span>
                        <span
                          className="px-1.5 py-0.2 rounded text-[10px] uppercase
                                     font-bold bg-slate-800 text-slate-300"
                        >
                          {edge.type}
                        </span>
                        <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                        <span className="text-emerald-400">
                          mem #{edge.dst_memory_id}
                        </span>
                      </div>
                      {edge.reason && (
                        <span className="text-[11px] text-slate-400 truncate max-w-xs">
                          {edge.reason}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
