import React from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Box } from 'lucide-react';

export interface EntityNodeData extends Record<string, unknown> {
  id: string;
  label: string;
  meta?: Record<string, unknown>;
}

export function EntityNode({ data, selected }: NodeProps): React.JSX.Element {
  const nodeData = data as unknown as EntityNodeData;
  const containerClass = selected
    ? 'border-indigo-400 ring-2 ring-indigo-400/50 shadow-indigo-950/50'
    : 'border-indigo-700/60 hover:border-indigo-500';

  const baseClass =
    'px-4 py-3 rounded-xl bg-slate-900/95 border min-w-[180px] shadow-lg transition-all';
  const iconBoxClass =
    'p-2 rounded-lg bg-indigo-950/80 text-indigo-400 border border-indigo-800/60';

  return (
    <div className={`${baseClass} ${containerClass}`}>
      <Handle
        type="target"
        position={Position.Left}
        className="w-2.5 h-2.5 bg-indigo-500 border border-slate-900"
      />
      <div className="flex items-center gap-2.5">
        <div className={iconBoxClass}>
          <Box className="w-4 h-4" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400">
              Entity
            </span>
          </div>
          <div
            className="text-sm font-semibold text-slate-100 truncate"
            title={nodeData.label}
          >
            {nodeData.label}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        className="w-2.5 h-2.5 bg-indigo-500 border border-slate-900"
      />
    </div>
  );
}
