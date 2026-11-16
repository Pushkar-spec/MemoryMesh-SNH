import React from 'react';
import { formatConfidence } from '../../lib/format';

interface ConfidenceBarProps {
  score: number | null | undefined;
  showLabel?: boolean;
  className?: string;
}

export function ConfidenceBar({
  score,
  showLabel = true,
  className = '',
}: ConfidenceBarProps): React.JSX.Element {
  const safeScore = score !== null && score !== undefined && !isNaN(score) ? score : 0;
  const clamped = Math.max(0, Math.min(1, safeScore));
  const percentage = Math.round(clamped * 100);

  let barColor = 'bg-rose-500';
  let textColor = 'text-rose-400';

  if (clamped >= 0.8) {
    barColor = 'bg-emerald-500';
    textColor = 'text-emerald-400';
  } else if (clamped >= 0.5) {
    barColor = 'bg-amber-500';
    textColor = 'text-amber-400';
  }

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className="flex-1 bg-slate-800 rounded-full h-2 min-w-[60px] overflow-hidden border border-slate-700/50">
        <div
          className={`h-full rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <span className={`text-xs font-mono font-semibold min-w-[42px] text-right ${textColor}`}>
          {formatConfidence(score)}
        </span>
      )}
    </div>
  );
}
