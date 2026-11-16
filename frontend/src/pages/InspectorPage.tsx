import React from 'react';
import { Terminal } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function InspectorPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">LLM Inspector</h2>
        <p className="text-xs text-slate-400 mt-1">
          Inspect raw LLM provider calls, prompt templates, completions, latencies, and parsed structured outputs.
        </p>
      </div>

      <EmptyState
        icon={Terminal}
        title="LLM Inspector Ready for P9.7"
        description="LLM provider call auditing, latency monitoring, and token tracking will be integrated in batch P9.7."
      />
    </div>
  );
}
