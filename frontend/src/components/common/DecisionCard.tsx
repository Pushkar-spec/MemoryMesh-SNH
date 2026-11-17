import React from 'react';
import {
  CheckCircle,
  RefreshCw,
  PlusCircle,
  AlertTriangle,
  XCircle,
  LucideIcon,
} from 'lucide-react';
import type { ResolutionDecision } from '../../api/client';
import { formatUtcDateTime } from '../../lib/format';

interface DecisionCardProps {
  decision: ResolutionDecision;
  onMemoryClick?: (memoryId: number) => void;
  className?: string;
}

export function DecisionCard({
  decision,
  onMemoryClick,
  className = '',
}: DecisionCardProps): React.JSX.Element {
  const action = (decision.action || 'UNKNOWN').toUpperCase();

  let ActionIcon: LucideIcon = PlusCircle;
  let actionColor = 'text-slate-400 bg-slate-800/40 border-slate-700';

  switch (action) {
    case 'INSERT':
      ActionIcon = PlusCircle;
      actionColor = 'text-emerald-400 bg-emerald-950/40 border-emerald-800/50';
      break;
    case 'REPLACE':
      ActionIcon = RefreshCw;
      actionColor = 'text-amber-400 bg-amber-950/40 border-amber-800/50';
      break;
    case 'CORROBORATE':
      ActionIcon = CheckCircle;
      actionColor = 'text-sky-400 bg-sky-950/40 border-sky-800/50';
      break;
    case 'DISPUTE':
      ActionIcon = AlertTriangle;
      actionColor = 'text-rose-400 bg-rose-950/40 border-rose-800/50';
      break;
    case 'REJECT':
      ActionIcon = XCircle;
      actionColor = 'text-slate-400 bg-slate-900/60 border-slate-800';
      break;
    default:
      ActionIcon = PlusCircle;
      actionColor = 'text-slate-400 bg-slate-800/40 border-slate-700';
      break;
  }

  // Extract claim details if available
  const claimJson = decision.claim_json as Record<string, unknown> | null;
  const subject =
    (claimJson?.subject as string) || (claimJson?.entity as string) || '';
  const attribute = (claimJson?.attribute as string) || '';
  const value =
    (claimJson?.value as string) || (claimJson?.value_norm as string) || '';

  return (
    <div
      className={`p-4 rounded-xl border border-slate-800 bg-slate-900/40
                 hover:bg-slate-900/60 transition space-y-3 ${className}`}
    >
      {/* Header: Action & Rule Badges */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <span
            className={`inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full
                       text-xs font-bold uppercase tracking-wider border ${actionColor}`}
          >
            <ActionIcon className="w-3.5 h-3.5" />
            <span>{action}</span>
          </span>

          {decision.rule_id && (
            <span
              className="px-2 py-0.5 rounded text-[11px] font-mono font-bold
                         bg-slate-800 text-slate-300 border border-slate-700"
            >
              {decision.rule_id}
            </span>
          )}
        </div>

        <span className="text-[11px] font-mono text-slate-500">
          {formatUtcDateTime(decision.created_at)}
        </span>
      </div>

      {/* Target Claim / Memory info */}
      {(subject || attribute || value || decision.memory_id) && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {subject && (
            <span
              className="px-2 py-0.5 rounded bg-slate-950 text-slate-300
                         font-semibold border border-slate-800"
            >
              {subject}
            </span>
          )}
          {attribute && (
            <span
              className="px-2 py-0.5 rounded bg-slate-950 text-indigo-300
                         font-mono border border-slate-800"
            >
              {attribute}
            </span>
          )}
          {value && (
            <span
              className="px-2 py-0.5 rounded bg-slate-950 text-emerald-300
                         font-mono font-bold border border-slate-800"
            >
              = &quot;{value}&quot;
            </span>
          )}

          {decision.memory_id && (
            <button
              type="button"
              onClick={() => onMemoryClick && onMemoryClick(decision.memory_id!)}
              className={`ml-auto px-2 py-0.5 rounded text-[11px] font-mono
                         text-indigo-400 bg-indigo-950/60 border border-indigo-800/60 ${
                           onMemoryClick
                             ? 'cursor-pointer hover:bg-indigo-900/80 hover:text-indigo-200'
                             : ''
                         }`}
            >
              mem #{decision.memory_id}
            </button>
          )}
        </div>
      )}

      {/* Explanation prose from engine */}
      {decision.explanation && (
        <p
          className="text-xs text-slate-300 leading-relaxed bg-slate-950/50
                     p-2.5 rounded-lg border border-slate-800/60"
        >
          {decision.explanation}
        </p>
      )}

      {/* Scores breakdown pills */}
      {decision.scores && Object.keys(decision.scores).length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 pt-1 text-[11px] font-mono">
          <span className="text-slate-500 text-[10px] uppercase font-sans font-semibold">
            Scores:
          </span>
          {Object.entries(decision.scores).map(([k, v]) => (
            <span
              key={k}
              className="px-2 py-0.5 rounded bg-slate-950 text-slate-400 border border-slate-800/80"
            >
              {k}:{' '}
              <span className="text-slate-200 font-semibold">
                {typeof v === 'number' ? v.toFixed(3) : String(v)}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
