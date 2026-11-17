import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  FileQuestion,
  Database,
  ExternalLink,
  ShieldAlert,
} from 'lucide-react';
import type { AskResult } from '../../api/client';

interface AnswerCardProps {
  result: AskResult;
  onMemoryClick?: (memoryId: number) => void;
  className?: string;
}

export function AnswerCard({
  result,
  onMemoryClick,
  className = '',
}: AnswerCardProps): React.JSX.Element {
  const ans = result.answer;
  const isAbstained = ans.abstained;
  const isConflict = ans.conflict_flagged;
  const isNeedsReview = ans.needs_review_flagged;

  const answerDisplay =
    ans.value ||
    (ans.values && ans.values.length > 0 ? ans.values.join(', ') : null) ||
    ans.answer_text;

  return (
    <div
      className={`p-6 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4 ${className}`}
    >
      {/* Header & Status Indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div
            className="p-1.5 rounded-lg bg-indigo-600/30 border border-indigo-500/40
                       text-indigo-400"
          >
            <CheckCircle2 className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-white tracking-tight">
            MemoryMesh Answer
          </h3>
        </div>

        <span
          className="text-[11px] font-mono text-emerald-400 px-2 py-0.5 rounded
                     bg-emerald-950/60 border border-emerald-800/60"
        >
          Conflict-Aware Grounded
        </span>
      </div>

      {/* Conflict / Review / Abstention Banners */}
      {isAbstained && (
        <div
          className="p-3.5 rounded-lg border border-amber-800/80 bg-amber-950/40
                     text-amber-200 text-xs flex items-start space-x-2.5"
        >
          <FileQuestion className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-amber-300">Answer Abstained</span>
            <p className="mt-0.5 text-amber-200/90 leading-relaxed">
              MemoryMesh withheld a definitive factual answer due to unresolved conflicts
              or insufficient grounded evidence.
            </p>
          </div>
        </div>
      )}

      {isConflict && !isAbstained && (
        <div
          className="p-3 rounded-lg border border-amber-800/80 bg-amber-950/30
                     text-amber-200 text-xs flex items-center space-x-2"
        >
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            Notice: Conflicting assertions detected in memory graph for this query slot.
          </span>
        </div>
      )}

      {isNeedsReview && (
        <div
          className="p-3 rounded-lg border border-rose-800/80 bg-rose-950/30
                     text-rose-200 text-xs flex items-center space-x-2"
        >
          <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
          <span>
            Notice: One or more underlying cited memories are flagged for human review.
          </span>
        </div>
      )}

      {/* Primary Answer Display */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70">
        <p className="text-base text-white font-medium leading-relaxed">
          {answerDisplay || (
            <span className="italic text-slate-500">
              No direct factual answer was composed.
            </span>
          )}
        </p>
      </div>

      {/* Explanation prose from resolution engine */}
      {ans.explanation && (
        <div className="text-xs text-slate-300 space-y-1">
          <span
            className="font-semibold text-slate-400 uppercase tracking-wider text-[10px]"
          >
            Resolution Explanation:
          </span>
          <p
            className="bg-slate-950/40 p-3 rounded-lg border border-slate-800/60
                       font-mono text-[11px] leading-relaxed text-slate-300"
          >
            {ans.explanation}
          </p>
        </div>
      )}

      {/* Cited Memories & Provenance Links */}
      {ans.cited_memory_ids && ans.cited_memory_ids.length > 0 && (
        <div className="pt-2 border-t border-slate-800/80 space-y-2">
          <span
            className="text-[11px] font-semibold text-slate-400 uppercase
                       tracking-wider flex items-center space-x-1.5"
          >
            <Database className="w-3.5 h-3.5 text-indigo-400" />
            <span>Cited Memory Sources ({ans.cited_memory_ids.length})</span>
          </span>

          <div className="flex flex-wrap gap-2">
            {ans.cited_memory_ids.map((id: number) => (
              <button
                key={id}
                type="button"
                onClick={() => onMemoryClick && onMemoryClick(id)}
                className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg
                           text-xs font-mono font-medium bg-indigo-950/60 hover:bg-indigo-900/80
                           text-indigo-300 border border-indigo-800/80 transition"
              >
                <span>Memory #{id}</span>
                <ExternalLink className="w-3 h-3 opacity-60" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
