import React from 'react';
import { UploadCloud } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function IngestPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">Ingest Messages</h2>
        <p className="text-xs text-slate-400 mt-1">
          Extract claims, resolve entities and attributes, and execute conflict-aware resolution rules.
        </p>
      </div>

      <EmptyState
        icon={UploadCloud}
        title="Ingest Interface Ready for P9.3"
        description="The ingest form, structured/unstructured claim extraction, and Decision Cards will be integrated in batch P9.3."
      />
    </div>
  );
}
