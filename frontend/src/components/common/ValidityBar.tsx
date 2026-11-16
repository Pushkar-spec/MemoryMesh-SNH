import React from 'react';
import { Calendar } from 'lucide-react';
import { formatValidityInterval } from '../../lib/format';

interface ValidityBarProps {
  validFrom: string | null | undefined;
  validTo: string | null | undefined;
  className?: string;
}

export function ValidityBar({
  validFrom,
  validTo,
  className = '',
}: ValidityBarProps): React.JSX.Element {
  const isOpenInterval = !validTo;

  return (
    <div
      className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-xs text-slate-300 ${className}`}
    >
      <Calendar className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
      <span className="font-mono">{formatValidityInterval(validFrom, validTo)}</span>
      {isOpenInterval ? (
        <span className="ml-1 px-1.5 py-0.2 rounded text-[10px] uppercase font-bold tracking-wider bg-emerald-950 text-emerald-400 border border-emerald-800/60">
          Open
        </span>
      ) : (
        <span className="ml-1 px-1.5 py-0.2 rounded text-[10px] uppercase font-bold tracking-wider bg-slate-800 text-slate-400 border border-slate-700">
          Closed
        </span>
      )}
    </div>
  );
}
