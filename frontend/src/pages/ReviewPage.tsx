import React from 'react';
import { Scale } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function ReviewPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">Disputes & Review</h2>
        <p className="text-xs text-slate-400 mt-1">
          Review flagged memories, inspect conflicting claims, select winners, and acknowledge human review.
        </p>
      </div>

      <EmptyState
        icon={Scale}
        title="Disputes & Review Ready for P9.5"
        description="Dispute group resolution and human review acknowledgement flows will be integrated in batch P9.5."
      />
    </div>
  );
}
