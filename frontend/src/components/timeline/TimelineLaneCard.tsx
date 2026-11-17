import React from 'react';
import {
  Clock,
  Calendar,
  XCircle,
  ExternalLink,
  AlertTriangle,
  History,
} from 'lucide-react';
import type { SlotTimeline, MemoryView } from '../../api/client';
import { StatusBadge } from '../common/StatusBadge';
import { ConfidenceBar } from '../common/ConfidenceBar';
import { ValidityBar } from '../common/ValidityBar';
import { formatUtcDateTime } from '../../lib/format';
import { getTimelineLaneStats } from '../../lib/timeline';

interface TimelineLaneCardProps {
  lane: SlotTimeline;
  onSelectMemory: (id: number) => void;
}

export const TimelineLaneCard: React.FC<TimelineLaneCardProps> = ({
  lane,
  onSelectMemory,
}) => {
  const memories = lane.memories ?? [];
  const rejected = lane.rejected ?? [];

  const {
    activeCount,
    historicalCount,
    disputedCount,
    rejectedCount,
  } = getTimelineLaneStats(lane);

  const headerIconClass =
    'p-2 rounded-lg bg-indigo-950/80 text-indigo-400 border border-indigo-800/60';
  const attrBadgeClass =
    'px-2 py-0.5 rounded text-xs font-semibold bg-indigo-900/50 ' +
    'text-indigo-300 border border-indigo-700/50';
  const ctxBadgeClass =
    'px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 ' +
    'text-slate-300 border border-slate-700';

  const activePillClass =
    'px-2 py-0.5 rounded-full bg-emerald-950/60 text-emerald-300 ' +
    'border border-emerald-800/80 font-medium';
  const histPillClass =
    'px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 font-medium';
  const disputedPillClass =
    'px-2 py-0.5 rounded-full bg-amber-950/60 text-amber-300 ' +
    'border border-amber-800/80 font-medium';
  const rejPillClass =
    'px-2 py-0.5 rounded-full bg-rose-950/60 text-rose-300 border border-rose-800/80 font-medium';

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-5 shadow-lg">
      {/* Lane Header */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800"
      >
        <div className="flex items-center gap-2.5">
          <div className={headerIconClass}>
            <History className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base text-white">{lane.entity}</span>
              <span className={attrBadgeClass}>{lane.attribute}</span>
              {lane.context_key && (
                <span className={ctxBadgeClass}>{lane.context_key}</span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Temporal slot lane for entity attribute changes and validity intervals.
            </p>
          </div>
        </div>

        {/* Counter Pills */}
        <div className="flex items-center gap-2 text-xs">
          {activeCount > 0 && <span className={activePillClass}>{activeCount} active</span>}
          {historicalCount > 0 && (
            <span className={histPillClass}>
              {historicalCount} historical
            </span>
          )}
          {disputedCount > 0 && (
            <span className={disputedPillClass}>
              {disputedCount} disputed
            </span>
          )}
          {rejectedCount > 0 && (
            <span className={rejPillClass}>{rejectedCount} rejected</span>
          )}
        </div>
      </div>

      {/* Accepted Memory Evolution */}
      <div className="space-y-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
          Memory Evolution ({memories.length})
        </span>

        {memories.length === 0 ? (
          <div className="text-xs text-slate-400 italic py-2">
            No accepted memories in this slot lane.
          </div>
        ) : (
          <div className="space-y-3">
            {memories.map((mv: MemoryView) => {
              const mem = mv.memory;
              const cardBg =
                mem.status === 'active'
                  ? 'bg-slate-900/90 border-blue-500/40 shadow-sm'
                  : 'bg-slate-950/70 border-slate-800/80';
              const valClass =
                'font-mono text-sm font-bold text-slate-100 bg-slate-950 ' +
                'px-2.5 py-0.5 rounded border border-slate-800';

              return (
                <div key={mem.id} className={`p-4 rounded-xl border transition-all ${cardBg}`}>
                  <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-slate-400">
                        #{mem.id}
                      </span>
                      <StatusBadge
                        status={mem.status}
                        needsReview={mem.needs_review}
                      />
                      <span className={valClass}>
                        {mem.value_display || mem.value_norm}
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={() => onSelectMemory(mem.id)}
                      className={
                        'flex items-center gap-1 text-xs font-medium text-blue-400 ' +
                        'hover:text-blue-300 transition-colors cursor-pointer'
                      }
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                      Provenance
                    </button>
                  </div>

                  {/* Dual-Time & Temporal Information */}
                  <div
                    className={
                      'grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-2 ' +
                      'border-t border-slate-800/60'
                    }
                  >
                    {/* Bitemporal Dimension 1: Valid Time */}
                    <div className="space-y-1">
                      <span
                        className="text-slate-400 font-semibold flex items-center gap-1 text-[11px]"
                      >
                        <Calendar className="w-3.5 h-3.5 text-indigo-400" />
                        Valid Time (World Time):
                      </span>
                      <ValidityBar validFrom={mem.valid_from} validTo={mem.valid_to} />
                    </div>

                    {/* Bitemporal Dimension 2: Recorded / Belief Time */}
                    <div className="space-y-1">
                      <span
                        className="text-slate-400 font-semibold flex items-center gap-1 text-[11px]"
                      >
                        <Clock className="w-3.5 h-3.5 text-blue-400" />
                        Recorded Time (System Time):
                      </span>
                      <div
                        className={
                          'font-mono text-[11px] text-slate-300 bg-slate-950 px-2 ' +
                          'py-1 rounded border border-slate-800 inline-block'
                        }
                      >
                        Learned: {formatUtcDateTime(mem.recorded_at)}
                        {mem.closed_at && (
                          <span className="text-slate-400 block text-[10px]">
                            Closed: {formatUtcDateTime(mem.closed_at)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Confidence & Source */}
                  <div
                    className={
                      'flex flex-wrap items-center justify-between gap-3 text-xs ' +
                      'text-slate-400 pt-3 mt-2 border-t border-slate-800/40'
                    }
                  >
                    <div className="flex items-center gap-2 w-48">
                      <span className="shrink-0 text-[11px]">Confidence:</span>
                      <ConfidenceBar score={mv.effective_confidence} />
                    </div>

                    <div className="flex items-center gap-3 text-[11px]">
                      {mv.source_names && mv.source_names.length > 0 && (
                        <span>
                          Source:{' '}
                          <span className="text-slate-300 font-mono">
                            {mv.source_names.join(', ')}
                          </span>
                        </span>
                      )}
                      {mem.review_reason && (
                        <span className="text-amber-400 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          {mem.review_reason}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Rejected Claims Section (with Rejected-Memory Markers) */}
      {rejected.length > 0 && (
        <div className="pt-3 border-t border-slate-800 space-y-3">
          <div className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-rose-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-rose-300">
              Rejected Memory Markers ({rejected.length})
            </span>
          </div>

          <div className="space-y-2">
            {rejected.map((rej: MemoryView) => {
              const mem = rej.memory;
              const rejBadgeClass =
                'px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ' +
                'bg-rose-950 text-rose-300 border border-rose-800/80';
              return (
                <div
                  key={mem.id}
                  className={
                    'p-3 rounded-lg bg-rose-950/20 border border-rose-900/40 flex ' +
                    'flex-wrap items-center justify-between gap-3 text-xs'
                  }
                >
                  <div className="flex items-center gap-2.5">
                    <span className={rejBadgeClass}>Rejected</span>
                    <span className="font-mono font-semibold text-rose-200">
                      {mem.value_display || mem.value_norm}
                    </span>
                    <span className="text-slate-400 text-[11px]">
                      via {mem.source_id || 'unknown'} at{' '}
                      {formatUtcDateTime(mem.valid_from)}
                    </span>
                  </div>

                  {mem.review_reason && (
                    <span className="text-rose-400/90 text-xs italic">
                      Reason: {mem.review_reason}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
