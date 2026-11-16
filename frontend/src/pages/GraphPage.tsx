import React from 'react';
import { Share2 } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function GraphPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">Dependency Graph</h2>
        <p className="text-xs text-slate-400 mt-1">
          Interactive React Flow visualization of memories, derivations, disputes, and supersede relations.
        </p>
      </div>

      <EmptyState
        icon={Share2}
        title="Dependency Graph Ready for P9.4"
        description="Interactive React Flow graph with entity and memory nodes will be integrated in batch P9.4."
      />
    </div>
  );
}
