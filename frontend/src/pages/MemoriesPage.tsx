import React from 'react';
import { Database } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function MemoriesPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">Memory Browser</h2>
        <p className="text-xs text-slate-400 mt-1">
          Explore resolved memories, filter by entity, attribute, status, and inspect temporal intervals.
        </p>
      </div>

      <EmptyState
        icon={Database}
        title="Memories Browser Ready for P9.3"
        description="Memory listing, filtering, pagination, and provenance drawer will be integrated in batch P9.3."
      />
    </div>
  );
}
