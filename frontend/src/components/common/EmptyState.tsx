import React from 'react';
import { LucideIcon, Inbox } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function EmptyState({
  icon: Icon = Inbox,
  title,
  description,
  action,
  className = '',
}: EmptyStateProps): React.JSX.Element {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center rounded-xl border border-dashed border-slate-800 bg-slate-900/30 ${className}`}
    >
      <div className="p-3 bg-slate-800/80 rounded-full text-slate-400 mb-3 border border-slate-700/50">
        <Icon className="w-6 h-6 text-indigo-400" />
      </div>
      <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
      {description && (
        <p className="mt-1 text-xs text-slate-400 max-w-sm">{description}</p>
      )}
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="mt-4 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
