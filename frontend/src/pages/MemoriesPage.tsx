import React, { useState } from 'react';
import {
  Database,
  Search,
  Filter,
  RefreshCw,
  ExternalLink,
  Layers,
  X,
} from 'lucide-react';
import { useMemories, useCatalog } from '../api/hooks';
import type { MemoryStatus } from '../api/client';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';
import { StatusBadge } from '../components/common/StatusBadge';
import { ConfidenceBar } from '../components/common/ConfidenceBar';
import { ValidityBar } from '../components/common/ValidityBar';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';

export function MemoriesPage(): React.JSX.Element {
  // Filter state
  const [entityFilter, setEntityFilter] = useState('');
  const [attributeFilter, setAttributeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false);

  // Provenance drawer state
  const [selectedMemoryId, setSelectedMemoryId] = useState<number | null>(null);

  // Queries
  const { data: catalog } = useCatalog();
  const {
    data: memories,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useMemories({
    entity: entityFilter.trim() || undefined,
    attribute: attributeFilter.trim() || undefined,
    status: (statusFilter.trim() as MemoryStatus) || undefined,
    needs_review: needsReviewOnly ? true : undefined,
  });

  const handleResetFilters = () => {
    setEntityFilter('');
    setAttributeFilter('');
    setStatusFilter('');
    setNeedsReviewOnly(false);
  };

  const hasActiveFilters = Boolean(
    entityFilter.trim() || attributeFilter.trim() || statusFilter || needsReviewOnly
  );

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Page Header */}
      <div
        className="border-b border-slate-800 pb-4 flex flex-col sm:flex-row
                   sm:items-center justify-between gap-3"
      >
        <div>
          <h2 className="text-lg font-bold text-white tracking-tight flex items-center space-x-2">
            <Database className="w-5 h-5 text-indigo-400" />
            <span>Persisted Memories</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Browse resolved, conflict-aware memory records, inspect temporal validity intervals,
            and review provenance.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg
                       bg-slate-900 hover:bg-slate-800 border border-slate-800
                       text-xs font-medium text-slate-300 transition"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin text-indigo-400' : ''}`}
            />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 space-y-3">
        <div
          className="flex items-center justify-between text-xs font-semibold
                     text-slate-400 uppercase tracking-wider"
        >
          <span className="flex items-center space-x-1.5">
            <Filter className="w-4 h-4 text-indigo-400" />
            <span>Query Filters</span>
          </span>
          {hasActiveFilters && (
            <button
              type="button"
              onClick={handleResetFilters}
              className="text-[11px] text-indigo-400 hover:underline flex items-center
                         space-x-1 normal-case"
            >
              <X className="w-3 h-3" />
              <span>Clear Filters</span>
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          {/* Entity Filter */}
          <div>
            <label className="block text-slate-400 font-medium mb-1">
              Entity Name
            </label>
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
              <input
                type="text"
                value={entityFilter}
                onChange={(e) => setEntityFilter(e.target.value)}
                placeholder="e.g. Server 1"
                className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800
                           text-slate-200 text-xs focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          {/* Attribute Dropdown */}
          <div>
            <label className="block text-slate-400 font-medium mb-1">
              Attribute
            </label>
            <select
              value={attributeFilter}
              onChange={(e) => setAttributeFilter(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800
                         text-slate-200 text-xs focus:outline-none focus:border-indigo-500
                         font-mono"
            >
              <option value="">All Attributes</option>
              {catalog?.attributes?.map((attr) => (
                <option key={attr.attribute} value={attr.attribute}>
                  {attr.attribute}
                </option>
              ))}
            </select>
          </div>

          {/* Status Dropdown */}
          <div>
            <label className="block text-slate-400 font-medium mb-1">
              Memory Status
            </label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800
                         text-slate-200 text-xs focus:outline-none focus:border-indigo-500"
            >
              <option value="">All Statuses</option>
              <option value="active">Active</option>
              <option value="historical">Historical</option>
              <option value="disputed">Disputed</option>
              <option value="superseded">Superseded</option>
            </select>
          </div>

          {/* Needs Review Toggle */}
          <div className="flex items-center pt-5">
            <label
              className="inline-flex items-center space-x-2 cursor-pointer
                         text-xs text-slate-300"
            >
              <input
                type="checkbox"
                checked={needsReviewOnly}
                onChange={(e) => setNeedsReviewOnly(e.target.checked)}
                className="w-4 h-4 rounded bg-slate-950 border-slate-800 text-indigo-600
                           focus:ring-0 focus:ring-offset-0"
              />
              <span>Needs Review Only</span>
            </label>
          </div>
        </div>
      </div>

      {/* Memory Table / Cards Content */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/30 overflow-hidden">
        {isLoading ? (
          <div className="py-20 flex justify-center">
            <LoadingSpinner size="lg" label="Loading memories from repository..." />
          </div>
        ) : error ? (
          <div className="p-6">
            <ErrorState
              error={error}
              title="Failed to load memories"
              onRetry={() => refetch()}
            />
          </div>
        ) : !memories || memories.length === 0 ? (
          <div className="p-8">
            <EmptyState
              icon={Database}
              title="No Memories Found"
              description={
                hasActiveFilters
                  ? 'No memories match the active filter criteria. Try clearing filters.'
                  : 'The repository is currently empty. Use the Ingest page to add messages.'
              }
              action={
                hasActiveFilters
                  ? { label: 'Clear Filters', onClick: handleResetFilters }
                  : undefined
              }
            />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300 border-collapse">
              <thead>
                <tr
                  className="border-b border-slate-800/80 bg-slate-950/60 text-[11px]
                             font-semibold uppercase tracking-wider text-slate-400"
                >
                  <th className="py-3 px-4">ID</th>
                  <th className="py-3 px-4">Subject (Entity)</th>
                  <th className="py-3 px-4">Attribute</th>
                  <th className="py-3 px-4">Value</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Validity Period</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {memories.map((view) => {
                  const m = view.memory;
                  return (
                    <tr
                      key={m.id}
                      className="hover:bg-slate-800/40 transition group"
                    >
                      {/* ID */}
                      <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                        #{m.id}
                      </td>

                      {/* Subject */}
                      <td className="py-3 px-4 font-medium text-white">
                        {m.entity_name || `Entity #${m.entity_id}`}
                      </td>

                      {/* Attribute */}
                      <td className="py-3 px-4 font-mono text-indigo-300">
                        {m.attribute}
                      </td>

                      {/* Value & context */}
                      <td className="py-3 px-4 font-mono font-bold text-emerald-400">
                        <span>{m.value_display || m.value_norm}</span>
                        {m.context && Object.keys(m.context).length > 0 && (
                          <div className="text-[10px] text-slate-400 font-normal font-sans">
                            {Object.entries(m.context)
                              .map(([k, v]) => `${k}=${String(v)}`)
                              .join(', ')}
                          </div>
                        )}
                      </td>

                      {/* Status */}
                      <td className="py-3 px-4">
                        <StatusBadge
                          status={m.status}
                          needsReview={m.needs_review}
                        />
                      </td>

                      {/* Confidence */}
                      <td className="py-3 px-4 min-w-[120px]">
                        <ConfidenceBar
                          score={view.effective_confidence ?? m.confidence}
                        />
                      </td>

                      {/* Validity Period */}
                      <td className="py-3 px-4">
                        <ValidityBar
                          validFrom={m.valid_from}
                          validTo={m.valid_to}
                        />
                      </td>

                      {/* Action */}
                      <td className="py-3 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => setSelectedMemoryId(m.id)}
                          className="inline-flex items-center space-x-1 px-2.5 py-1 rounded
                                     bg-slate-900 group-hover:bg-indigo-600 text-slate-300
                                     group-hover:text-white border border-slate-700/60
                                     text-[11px] font-medium transition"
                        >
                          <span>Provenance</span>
                          <ExternalLink className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {/* Footer Summary */}
            <div
              className="p-3 border-t border-slate-800 bg-slate-950/40 text-[11px]
                         text-slate-400 flex items-center justify-between font-mono"
            >
              <span className="flex items-center space-x-1.5">
                <Layers className="w-3.5 h-3.5 text-indigo-400" />
                <span>Showing {memories.length} memory records</span>
              </span>
              <span>MemoryMesh Engine V1</span>
            </div>
          </div>
        )}
      </div>

      {/* Slide-over Provenance Drawer */}
      <ProvenanceDrawer
        memoryId={selectedMemoryId}
        onClose={() => setSelectedMemoryId(null)}
        onSelectMemory={(id) => setSelectedMemoryId(id)}
      />
    </div>
  );
}
