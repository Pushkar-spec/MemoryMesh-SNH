import React from 'react';
import {
  AlertTriangle,
  ExternalLink,
  CheckCircle,
  RefreshCw,
  Clock,
} from 'lucide-react';
import type { MemoryView } from '../../api/client';
import { StatusBadge } from '../common/StatusBadge';
import { ConfidenceBar } from '../common/ConfidenceBar';
import { ValidityBar } from '../common/ValidityBar';
import { formatUtcDateTime } from '../../lib/format';

interface ReviewMemoryCardProps {
  memoryView: MemoryView;
  onSelectMemory: (id: number) => void;
  onAcknowledge: (memoryId: number) => Promise<void> | void;
  isAcknowledging?: boolean;
}

export function ReviewMemoryCard({
  memoryView,
  onSelectMemory,
  onAcknowledge,
  isAcknowledging = false,
}: ReviewMemoryCardProps): React.JSX.Element {
  const mem = memoryView.memory;

  const headerIconClass =
    'p-2 rounded-lg bg-indigo-950/80 text-indigo-400 border border-indigo-800/60';
  const attrBadgeClass =
    'px-2 py-0.5 rounded text-xs font-semibold bg-indigo-900/50 ' +
    'text-indigo-300 border border-indigo-700/50';
  const ctxBadgeClass =
    'px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 ' +
    'text-slate-300 border border-slate-700';
  const linkClass =
    'flex items-center gap-1 text-xs font-medium text-blue-400 ' +
    'hover:text-blue-300 cursor-pointer';
  const alertBoxClass =
    'flex items-start gap-2 p-2.5 rounded-lg bg-amber-950/25 ' +
    'border border-amber-900/40 text-xs text-amber-300';
  const valRowClass =
    'p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 flex ' +
    'flex-wrap items-center justify-between gap-2';
  const valBadgeClass =
    'font-mono text-sm font-bold text-emerald-300 bg-slate-900 ' +
    'px-2.5 py-0.5 rounded border border-slate-800';
  const ackBtnClass =
    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold ' +
    'bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-600/30 ' +
    'transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-lg">
      {/* Header */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800"
      >
        <div className="flex items-center gap-2.5">
          <div className={headerIconClass}>
            <Clock className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-slate-400">
                #{mem.id}
              </span>
              <span className="font-bold text-base text-white">{mem.entity_name}</span>
              <span className={attrBadgeClass}>{mem.attribute}</span>
              {mem.context_key && (
                <span className={ctxBadgeClass}>{mem.context_key}</span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Memory flagged for human review verification.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <StatusBadge status={mem.status} needsReview={mem.needs_review} />
          <button
            type="button"
            onClick={() => onSelectMemory(mem.id)}
            className={linkClass}
          >
            <ExternalLink className="w-3.5 h-3.5" />
            Provenance
          </button>
        </div>
      </div>

      {/* Review Reason Alert */}
      {mem.review_reason && (
        <div className={alertBoxClass}>
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span
              className="font-semibold block text-[11px] uppercase tracking-wider text-amber-400"
            >
              Flagged Reason:
            </span>
            <span>{mem.review_reason}</span>
          </div>
        </div>
      )}

      {/* Value Display */}
      <div className={valRowClass}>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Asserted Value:</span>
          <span className={valBadgeClass}>
            {mem.value_display || mem.value_norm}
          </span>
        </div>
        <div className="text-xs text-slate-400 flex items-center gap-2">
          <span>Source:</span>
          <span className="font-mono text-slate-200">{mem.source_id}</span>
        </div>
      </div>

      {/* Temporal and Confidence Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs pt-1">
        <div className="space-y-1.5">
          <span className="text-slate-400 font-semibold block text-[11px]">
            Validity Window:
          </span>
          <ValidityBar validFrom={mem.valid_from} validTo={mem.valid_to} />
          <span className="text-[11px] text-slate-500 block">
            Learned: {formatUtcDateTime(mem.recorded_at)}
          </span>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span className="font-semibold">Effective Confidence</span>
            {memoryView.is_stale && (
              <span className="text-rose-400 font-semibold">Stale (Decayed)</span>
            )}
          </div>
          <ConfidenceBar score={memoryView.effective_confidence} />
        </div>
      </div>

      {/* Action Footer */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800"
      >
        <p className="text-xs text-slate-400">
          Acknowledging will clear the review flag while preserving existing status and confidence.
        </p>

        <button
          type="button"
          disabled={isAcknowledging}
          onClick={() => onAcknowledge(mem.id)}
          className={ackBtnClass}
        >
          {isAcknowledging ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Acknowledging...</span>
            </>
          ) : (
            <>
              <CheckCircle className="w-3.5 h-3.5" />
              <span>Acknowledge Review</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
