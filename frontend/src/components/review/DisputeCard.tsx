import React, { useState } from 'react';
import {
  Scale,
  AlertTriangle,
  ExternalLink,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';
import type { DisputeView, MemoryView } from '../../api/client';
import { StatusBadge } from '../common/StatusBadge';
import { ConfidenceBar } from '../common/ConfidenceBar';
import { ValidityBar } from '../common/ValidityBar';
import { formatUtcDateTime } from '../../lib/format';

interface DisputeCardProps {
  dispute: DisputeView;
  onSelectMemory: (id: number) => void;
  onResolve: (winnerMemoryId: number) => Promise<void> | void;
  isResolving?: boolean;
}

export function DisputeCard({
  dispute,
  onSelectMemory,
  onResolve,
  isResolving = false,
}: DisputeCardProps): React.JSX.Element {
  const [selectedWinnerId, setSelectedWinnerId] = useState<number | null>(null);

  const memA = dispute.a.memory;
  const memB = dispute.b.memory;
  const entity = memA.entity_name || memB.entity_name || 'Unknown Entity';
  const attribute = memA.attribute || memB.attribute || 'Unknown Attribute';
  const contextKey = memA.context_key || memB.context_key || '';

  const renderCandidate = (mv: MemoryView) => {
    const mem = mv.memory;
    const isSelected = selectedWinnerId === mem.id;
    const borderStyle = isSelected
      ? 'border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50'
      : 'border-slate-800 bg-slate-950/60 hover:border-slate-700';

    const cardClass =
      `p-4 rounded-xl border transition-all cursor-pointer flex flex-col ${borderStyle}`;
    const valueClass =
      'font-mono text-sm font-bold text-white bg-slate-900 px-2 py-0.5 ' +
      'rounded border border-slate-800';
    const linkClass =
      'flex items-center gap-1 text-[11px] font-medium text-blue-400 hover:text-blue-300';

    return (
      <div
        key={mem.id}
        onClick={() => !isResolving && setSelectedWinnerId(mem.id)}
        className={cardClass}
      >
        {/* Top: Radio selection & ID */}
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/60 mb-3">
          <div className="flex items-center gap-2">
            <input
              type="radio"
              name={`winner-${memA.id}-${memB.id}`}
              checked={isSelected}
              onChange={() => setSelectedWinnerId(mem.id)}
              disabled={isResolving}
              className="accent-indigo-500 w-4 h-4 cursor-pointer"
            />
            <span className="font-mono text-xs font-bold text-slate-300">
              Memory #{mem.id}
            </span>
          </div>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectMemory(mem.id);
            }}
            className={linkClass}
          >
            <ExternalLink className="w-3 h-3" />
            Provenance
          </button>
        </div>

        {/* Claim Value & Status */}
        <div className="space-y-2 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={valueClass}>
              {mem.value_display || mem.value_norm}
            </span>
            <StatusBadge status={mem.status} needsReview={mem.needs_review} />
          </div>

          {/* Temporal & Source Details */}
          <div className="space-y-1.5 text-xs pt-1">
            <ValidityBar validFrom={mem.valid_from} validTo={mem.valid_to} />
            <div
              className={
                'flex flex-wrap items-center justify-between gap-1 text-[11px] text-slate-400'
              }
            >
              <span>Source: <span className="font-mono text-slate-300">{mem.source_id}</span></span>
              <span>Learned: {formatUtcDateTime(mem.recorded_at)}</span>
            </div>
          </div>
        </div>

        {/* Confidence score bar */}
        <div className="pt-3 mt-3 border-t border-slate-800/40">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
            <span>Effective Confidence</span>
          </div>
          <ConfidenceBar score={mv.effective_confidence} />
        </div>
      </div>
    );
  };

  const attrBadgeClass =
    'px-2 py-0.5 rounded text-xs font-semibold bg-indigo-900/50 ' +
    'text-indigo-300 border border-indigo-700/50';
  const ctxBadgeClass =
    'px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 ' +
    'text-slate-300 border border-slate-700';
  const disputeBadgeClass =
    'px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider ' +
    'bg-amber-950/60 text-amber-300 border border-amber-800/80';
  const reasonClass =
    'flex items-start gap-2 p-2.5 rounded-lg bg-amber-950/20 ' +
    'border border-amber-900/40 text-xs text-amber-300';

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-lg">
      {/* Dispute Header */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800"
      >
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-amber-950/80 text-amber-400 border border-amber-800/60">
            <Scale className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base text-white">{entity}</span>
              <span className={attrBadgeClass}>{attribute}</span>
              {contextKey && <span className={ctxBadgeClass}>{contextKey}</span>}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Conflicting active claims require human resolution.
            </p>
          </div>
        </div>

        <span className={disputeBadgeClass}>Dispute Active</span>
      </div>

      {/* Conflict Reason */}
      {dispute.edge_reason && (
        <div className={reasonClass}>
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <span>{dispute.edge_reason}</span>
        </div>
      )}

      {/* Competing Candidates Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {renderCandidate(dispute.a)}
        {renderCandidate(dispute.b)}
      </div>

      {/* Action Footer */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800"
      >
        <p className="text-xs text-slate-400">
          {selectedWinnerId
            ? `Memory #${selectedWinnerId} selected as winner. ` +
              'Submitting will supersede other claims.'
            : 'Select one of the competing claims above to resolve the dispute.'}
        </p>

        <button
          type="button"
          disabled={!selectedWinnerId || isResolving}
          onClick={() => selectedWinnerId && onResolve(selectedWinnerId)}
          className={
            'flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold ' +
            'transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ' +
            (selectedWinnerId
              ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/30'
              : 'bg-slate-800 text-slate-400 border border-slate-700')
          }
        >
          {isResolving ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Resolving...</span>
            </>
          ) : (
            <>
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>
                {selectedWinnerId ? `Confirm Winner (#${selectedWinnerId})` : 'Select Winner'}
              </span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
