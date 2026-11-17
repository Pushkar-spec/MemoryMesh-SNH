import React from 'react';
import { X, Box, ShieldCheck, ExternalLink, AlertTriangle } from 'lucide-react';
import type { GraphNode, GraphEdge } from '../../api/client';
import { StatusBadge } from '../common/StatusBadge';
import { ConfidenceBar } from '../common/ConfidenceBar';
import { ValidityBar } from '../common/ValidityBar';

interface GraphDetailPanelProps {
  node: GraphNode | null;
  edges: GraphEdge[];
  onClose: () => void;
  onInspectProvenance?: (memoryId: number) => void;
}

export const GraphDetailPanel: React.FC<GraphDetailPanelProps> = ({
  node,
  edges,
  onClose,
  onInspectProvenance,
}) => {
  if (!node) return null;

  const isEntity = node.kind === 'entity';
  const memoryId = !isEntity ? parseInt(node.id.replace(/^mem:/, ''), 10) : null;
  const connectedEdges = edges.filter(
    (e) => e.source === node.id || e.target === node.id
  );

  const closeBtnClass =
    'p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 ' +
    'transition-colors cursor-pointer';

  const inspectBtnClass =
    'w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs ' +
    'font-medium text-white bg-blue-600 hover:bg-blue-500 transition-colors shadow ' +
    'cursor-pointer';

  const panelClass =
    'w-80 sm:w-96 bg-slate-900 border-l border-slate-800 p-5 flex flex-col ' +
    'h-full overflow-y-auto space-y-5 shadow-2xl';

  return (
    <div className={panelClass}>
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          {isEntity ? (
            <div
              className="p-1.5 rounded-lg bg-indigo-950 text-indigo-400 border border-indigo-800"
            >
              <Box className="w-4 h-4" />
            </div>
          ) : (
            <div className="p-1.5 rounded-lg bg-blue-950 text-blue-400 border border-blue-800">
              <ShieldCheck className="w-4 h-4" />
            </div>
          )}
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              {isEntity ? 'Entity Details' : `Memory #${memoryId}`}
            </span>
            <h3
              className="text-sm font-bold text-white truncate max-w-[200px]"
              title={node.label}
            >
              {node.label}
            </h3>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className={closeBtnClass}
          aria-label="Close details"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Entity Node Details */}
      {isEntity && (
        <div className="space-y-4 text-xs">
          <div>
            <span className="text-slate-400 font-medium block mb-1">Entity Name:</span>
            <div className="text-sm font-semibold text-slate-100">{node.label}</div>
          </div>
          <div>
            <span className="text-slate-400 font-medium block mb-1">Entity ID:</span>
            <div className="font-mono text-slate-300">{node.id}</div>
          </div>
        </div>
      )}

      {/* Memory Node Details */}
      {!isEntity && (
        <div className="space-y-4 text-xs">
          {/* Status & Review Flag */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-slate-400 font-medium block mb-1">Status:</span>
              <StatusBadge status={node.status ?? 'active'} />
            </div>
            {node.needs_review && (
              <span
                className={
                  'flex items-center gap-1 text-[11px] font-medium text-amber-400 ' +
                  'bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/80'
                }
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                Needs Review
              </span>
            )}
          </div>

          {/* Entity & Attribute */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-slate-400 font-medium block mb-0.5">Entity:</span>
              <span className="font-semibold text-slate-200">
                {(node.meta?.entity as string) || '—'}
              </span>
            </div>
            <div>
              <span className="text-slate-400 font-medium block mb-0.5">Attribute:</span>
              <span className="font-semibold text-slate-200">
                {(node.meta?.attribute as string) || '—'}
              </span>
            </div>
          </div>

          {/* Value Display */}
          <div>
            <span className="text-slate-400 font-medium block mb-1">Stored Value:</span>
            <div
              className={
                'p-2.5 rounded-lg bg-slate-950 font-mono text-sm text-blue-300 ' +
                'border border-slate-800 break-all'
              }
            >
              {typeof node.meta?.value === 'string'
                ? node.meta.value
                : node.label}
            </div>
          </div>

          {/* Confidence */}
          <div>
            <span className="text-slate-400 font-medium block mb-1">Confidence Score:</span>
            <ConfidenceBar score={node.confidence ?? 0} />
          </div>

          {/* Validity Window */}
          <div>
            <span className="text-slate-400 font-medium block mb-1">Valid Interval:</span>
            <ValidityBar
              validFrom={node.meta?.valid_from as string | undefined}
              validTo={node.meta?.valid_to as string | undefined}
            />
          </div>

          {/* Source ID */}
          {Boolean(node.meta?.source_id) && (
            <div>
              <span className="text-slate-400 font-medium block mb-0.5">Source:</span>
              <span className="font-mono text-slate-300">
                {String(node.meta?.source_id)}
              </span>
            </div>
          )}

          {/* Inspect Provenance action */}
          {memoryId !== null && Boolean(onInspectProvenance) && (
            <div className="pt-2">
              <button
                type="button"
                onClick={() => onInspectProvenance?.(memoryId)}
                className={inspectBtnClass}
              >
                <ExternalLink className="w-3.5 h-3.5" />
                Inspect Full Provenance
              </button>
            </div>
          )}
        </div>
      )}

      {/* Connected Edges & Relationships */}
      <div className="pt-3 border-t border-slate-800 text-xs">
        <span className="text-slate-400 font-semibold block mb-2">
          Relationships ({connectedEdges.length}):
        </span>
        {connectedEdges.length === 0 ? (
          <div className="text-slate-400 text-xs italic">No connected relationships.</div>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {connectedEdges.map((e) => {
              const isSource = e.source === node.id;
              const otherNodeId = isSource ? e.target : e.source;
              const direction = isSource ? '→' : '←';
              return (
                <div
                  key={e.id}
                  className="p-2 rounded bg-slate-950/70 border border-slate-800/80 space-y-1"
                >
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-bold text-slate-300 uppercase tracking-wider">
                      {e.type}
                    </span>
                    <span className="font-mono text-slate-400">
                      {direction} {otherNodeId}
                    </span>
                  </div>
                  {e.reason && (
                    <p className="text-[11px] text-slate-400 leading-snug">{e.reason}</p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
