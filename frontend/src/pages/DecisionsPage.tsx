import React, { useState, useMemo } from 'react';
import {
  Activity,
  Filter,
  RefreshCw,
  X,
  Layers,
} from 'lucide-react';
import { useDecisions } from '../api/hooks';
import { extractAvailableRules } from '../lib/review';
import { DecisionCard } from '../components/common/DecisionCard';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';

const CANONICAL_RULES = [
  'R-INSERT',
  'R-REPLACE',
  'R-CORROBORATE',
  'R-DISPUTE',
  'R-DISPUTE-RESOLVE',
  'R-REJECT',
  'R-RETRACT',
] as const;

export function DecisionsPage(): React.JSX.Element {
  const [selectedRule, setSelectedRule] = useState<string>('');
  const [inspectMemoryId, setInspectMemoryId] = useState<number | null>(null);

  const queryParams = useMemo(() => {
    return selectedRule ? { rule_id: selectedRule } : undefined;
  }, [selectedRule]);

  const {
    data: decisionsData,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useDecisions(queryParams);

  const decisions = useMemo(() => decisionsData ?? [], [decisionsData]);

  // Available rules combined from canonical set and any dynamic rules in history
  const availableRules = useMemo(() => {
    return extractAvailableRules(decisions, CANONICAL_RULES);
  }, [decisions]);

  const filterBoxClass =
    'flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5';

  const clearBtnClass =
    'flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800 ' +
    'hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 ' +
    'transition-colors cursor-pointer';

  const refreshBtnClass =
    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 ' +
    'hover:bg-slate-700 text-slate-200 text-xs font-medium border ' +
    'border-slate-700 transition-colors cursor-pointer disabled:opacity-50';

  const badgeClass =
    'px-2 py-0.5 rounded-full text-xs font-semibold bg-slate-800 ' +
    'text-slate-300 border border-slate-700';

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div
        className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4"
      >
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">Decision Log</h2>
            {decisions.length > 0 && (
              <span className={badgeClass}>
                {decisions.length} {decisions.length === 1 ? 'record' : 'records'}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Complete audit trail of resolution decisions, rule firings, scores, and explanations.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Rule Selector */}
          <div className={filterBoxClass}>
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedRule}
              onChange={(e) => setSelectedRule(e.target.value)}
              className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="">All Rules</option>
              {availableRules.map((rule) => (
                <option key={rule} value={rule}>
                  {rule}
                </option>
              ))}
            </select>
          </div>

          {/* Reset Filter Button */}
          {selectedRule && (
            <button
              type="button"
              onClick={() => setSelectedRule('')}
              className={clearBtnClass}
              title="Clear rule filter"
            >
              <X className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          )}

          {/* Refresh Button */}
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className={refreshBtnClass}
            title="Refresh decision log"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Loading & Error States */}
      {isLoading && <LoadingSpinner label="Loading decision history..." />}

      {isError && (
        <ErrorState
          error={error}
          title="Failed to load decision log"
          onRetry={() => refetch()}
        />
      )}

      {/* Empty State */}
      {!isLoading && !isError && decisions.length === 0 && (
        <EmptyState
          icon={selectedRule ? Filter : Layers}
          title={selectedRule ? `No Decisions for ${selectedRule}` : 'No Decision Records'}
          description={
            selectedRule
              ? `No audit decisions matched rule filter '${selectedRule}'. ` +
                'Clear the filter to see all records.'
              : 'Ingest messages or resolve memory disputes to generate decision log entries.'
          }
          action={
            selectedRule
              ? {
                  label: 'Clear Rule Filter',
                  onClick: () => setSelectedRule(''),
                }
              : undefined
          }
        />
      )}

      {/* Decisions List */}
      {!isLoading && !isError && decisions.length > 0 && (
        <div className="space-y-4">
          {decisions.map((decision) => (
            <DecisionCard
              key={decision.id}
              decision={decision}
              onMemoryClick={(id) => setInspectMemoryId(id)}
            />
          ))}
        </div>
      )}

      {/* Provenance Drawer */}
      <ProvenanceDrawer
        memoryId={inspectMemoryId}
        onClose={() => setInspectMemoryId(null)}
      />
    </div>
  );
}
