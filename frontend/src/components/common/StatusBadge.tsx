import React from 'react';
import { AlertCircle, CheckCircle2, Clock, HelpCircle, XCircle } from 'lucide-react';

export type MemoryStatusType =
  | 'active'
  | 'historical'
  | 'disputed'
  | 'superseded'
  | string;

interface StatusBadgeProps {
  status: MemoryStatusType;
  needsReview?: boolean | number | null;
  className?: string;
}

export function StatusBadge({
  status,
  needsReview,
  className = '',
}: StatusBadgeProps): React.JSX.Element {
  const normStatus = status.toLowerCase();

  let badgeColor = 'bg-slate-800 text-slate-300 border-slate-700';
  let Icon = HelpCircle;

  switch (normStatus) {
    case 'active':
      badgeColor = 'bg-emerald-950/60 text-emerald-300 border-emerald-800/80';
      Icon = CheckCircle2;
      break;
    case 'historical':
      badgeColor = 'bg-slate-900/80 text-slate-400 border-slate-800';
      Icon = Clock;
      break;
    case 'disputed':
      badgeColor = 'bg-amber-950/60 text-amber-300 border-amber-800/80';
      Icon = AlertCircle;
      break;
    case 'superseded':
      badgeColor = 'bg-purple-950/60 text-purple-300 border-purple-800/80';
      Icon = XCircle;
      break;
  }

  const isReviewFlagged = Boolean(needsReview);

  return (
    <div className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${badgeColor}`}
      >
        <Icon className="w-3.5 h-3.5" />
        <span className="capitalize">{normStatus}</span>
      </span>

      {isReviewFlagged && (
        <span
          title="Flagged for human review"
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-950/70 text-rose-300 border border-rose-800/80"
        >
          <AlertCircle className="w-3 h-3" />
          <span>Needs Review</span>
        </span>
      )}
    </div>
  );
}
