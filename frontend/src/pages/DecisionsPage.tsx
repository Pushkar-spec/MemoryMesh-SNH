import React from 'react';
import { Activity } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function DecisionsPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">Decision Log</h2>
        <p className="text-xs text-slate-400 mt-1">
          Complete audit trail of all resolution decisions, rule firings, scores, and explanations.
        </p>
      </div>

      <EmptyState
        icon={Activity}
        title="Decision Log Ready for P9.5"
        description="Searchable, filterable decision audit log with full score breakdowns will be integrated in batch P9.5."
      />
    </div>
  );
}
