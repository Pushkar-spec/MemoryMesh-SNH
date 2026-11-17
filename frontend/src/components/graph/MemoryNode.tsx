import React from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { AlertTriangle, Clock } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';
import { formatConfidence, formatValidityInterval } from '../../lib/format';

export interface MemoryNodeData extends Record<string, unknown> {
  id: string;
  label: string;
  status?: string | null;
  confidence?: number | null;
  needs_review?: boolean;
  meta?: {
    value?: string;
    valid_from?: string;
    valid_to?: string | null;
    entity?: string;
    attribute?: string;
    source_id?: string;
    context_key?: string;
  };
}

export function MemoryNode({ data, selected }: NodeProps): React.JSX.Element {
  const nodeData = data as unknown as MemoryNodeData;
  const numericId = nodeData.id.replace(/^mem:/, '');
  const status = nodeData.status ?? 'active';
  const meta = nodeData.meta ?? {};

  const borderClass = selected
    ? 'border-blue-400 ring-2 ring-blue-400/50 shadow-blue-950/50'
    : 'border-slate-700 hover:border-slate-500';

  const reviewBadgeClass =
    'flex items-center gap-1 text-[10px] font-medium text-amber-400 ' +
    'bg-amber-950/50 px-1.5 py-0.5 rounded border border-amber-800/50';

  const cardBase =
    'p-3.5 rounded-xl bg-slate-900/95 border min-w-[240px] max-w-[280px] shadow-lg transition-all';

  return (
    <div className={`${cardBase} ${borderClass}`}>
      <Handle
        type="target"
        position={Position.Left}
        className="w-2.5 h-2.5 bg-slate-400 border border-slate-900"
      />

      {/* Header: ID, Status, and Review Alert */}
      <div className="flex items-center justify-between gap-2 mb-2 pb-2 border-b border-slate-800">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-xs font-semibold text-slate-400">
            #{numericId}
          </span>
          <StatusBadge status={status} />
        </div>
        {nodeData.needs_review && (
          <span className={reviewBadgeClass} title="Needs Review">
            <AlertTriangle className="w-3 h-3" />
            Review
          </span>
        )}
      </div>

      {/* Main label / Value */}
      <div className="mb-2">
        <div
          className="text-xs text-slate-400 font-medium truncate"
          title={meta.attribute}
        >
          {meta.attribute || 'Attribute'}
        </div>
        <div
          className="text-sm font-bold text-slate-100 truncate font-mono"
          title={meta.value || nodeData.label}
        >
          {meta.value || nodeData.label}
        </div>
      </div>

      {/* Footer: Confidence and Validity */}
      <div
        className={
          'flex items-center justify-between text-[11px] ' +
          'text-slate-400 pt-1 border-t border-slate-800/60'
        }
      >
        <div className="flex items-center gap-1">
          <span>Conf:</span>
          <span className="font-mono text-slate-200 font-medium">
            {formatConfidence(nodeData.confidence)}
          </span>
        </div>
        {(meta.valid_from || meta.valid_to) && (
          <div className="flex items-center gap-1 text-[10px] font-mono text-slate-400">
            <Clock className="w-3 h-3 text-slate-400" />
            <span>{formatValidityInterval(meta.valid_from, meta.valid_to)}</span>
          </div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="w-2.5 h-2.5 bg-slate-400 border border-slate-900"
      />
    </div>
  );
}
