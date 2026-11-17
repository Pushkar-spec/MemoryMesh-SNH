import React from 'react';
import { Bot, AlertCircle, FileText } from 'lucide-react';
import type { AskResponse } from '../../api/client';
import { formatUtcDateTime } from '../../lib/format';

interface NaiveRAGCompareCardProps {
  response: AskResponse;
  className?: string;
}

export function NaiveRAGCompareCard({
  response,
  className = '',
}: NaiveRAGCompareCardProps): React.JSX.Element | null {
  const baseline = response.baseline;
  const hits = response.baseline_retrieved ?? [];

  if (!baseline) return null;

  const baselineText =
    baseline.value ||
    (baseline.values && baseline.values.length > 0
      ? baseline.values.join(', ')
      : baseline.explanation || 'No answer provided by baseline.');

  return (
    <div
      className={`p-6 rounded-xl border border-slate-800 bg-slate-900/30 space-y-4 ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
            <Bot className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-200 tracking-tight">
            Naive RAG Baseline
          </h3>
        </div>

        <span
          className="text-[11px] font-mono text-amber-400 px-2 py-0.5 rounded
                     bg-amber-950/40 border border-amber-800/60"
        >
          Unaware of Conflicts & Validity
        </span>
      </div>

      {/* Warning Notice about Naive RAG limitations */}
      <div
        className="p-3 rounded-lg border border-slate-800 bg-slate-950/40 text-xs
                   text-slate-400 flex items-start space-x-2"
      >
        <AlertCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          The baseline retrieves text chunks purely via lexical or vector similarity without
          temporal intervals or conflict resolution, frequently surfacing superseded or
          conflicting claims.
        </p>
      </div>

      {/* Answer */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/50">
        <p className="text-sm text-slate-200 font-medium leading-relaxed font-serif">
          {baselineText}
        </p>
      </div>

      {/* Retrieved Hits Breakdown */}
      <div className="pt-2 border-t border-slate-800/80 space-y-2">
        <span
          className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider
                     flex items-center space-x-1.5"
        >
          <FileText className="w-3.5 h-3.5 text-slate-400" />
          <span>Retrieved Context Chunks ({hits.length})</span>
        </span>

        {hits.length > 0 ? (
          <div className="space-y-2">
            {hits.map((hit, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg border border-slate-800/80 bg-slate-950/30
                           text-xs space-y-1"
              >
                <div
                  className="flex items-center justify-between text-[11px]
                             text-slate-400 font-mono"
                >
                  <span className="font-semibold text-slate-300">
                    {hit.source_name || 'source'}
                  </span>
                  <span>
                    Score: {hit.score.toFixed(3)} &bull; {formatUtcDateTime(hit.event_time)}
                  </span>
                </div>
                <p className="text-slate-300 italic font-serif">&ldquo;{hit.text}&rdquo;</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic">No baseline context chunks retrieved.</p>
        )}
      </div>
    </div>
  );
}
