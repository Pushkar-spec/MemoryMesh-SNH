import React from 'react';
import { MessageSquare } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function AskPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">Ask Memory</h2>
        <p className="text-xs text-slate-400 mt-1">
          Query long-term memory with temporal reasoning, provenance citations, and side-by-side naive RAG comparison.
        </p>
      </div>

      <EmptyState
        icon={MessageSquare}
        title="Ask & Compare Ready for P9.3"
        description="Natural language question answering with naive RAG comparison will be integrated in batch P9.3."
      />
    </div>
  );
}
