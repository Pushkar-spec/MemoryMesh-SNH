import React from 'react';
import { Settings } from 'lucide-react';
import { EmptyState } from '../components/common/EmptyState';

export function SettingsPage(): React.JSX.Element {
  return (
    <div className="space-y-6">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight">System Settings</h2>
        <p className="text-xs text-slate-400 mt-1">
          Inspect engine thresholds, sources, attribute definitions, and active ablation flags.
        </p>
      </div>

      <EmptyState
        icon={Settings}
        title="Settings & Catalog Ready for P9.7"
        description="Configuration viewing, catalog inspection, and threshold settings will be integrated in batch P9.7."
      />
    </div>
  );
}
