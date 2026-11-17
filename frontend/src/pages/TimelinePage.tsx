import React, { useState } from 'react';
import { Clock, RefreshCw, Filter, Layers } from 'lucide-react';
import { useTimeline, useCatalog } from '../api/hooks';
import { getTimelineLaneKey } from '../lib/timeline';
import { TimelineLaneCard } from '../components/timeline/TimelineLaneCard';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';

export function TimelinePage(): React.JSX.Element {
  const [selectedEntity, setSelectedEntity] = useState<string>('');
  const [selectedAttribute, setSelectedAttribute] = useState<string>('');
  const [inspectMemoryId, setInspectMemoryId] = useState<number | null>(null);

  const { data: catalog } = useCatalog();

  const queryParams = {
    entity: selectedEntity || undefined,
    attribute: selectedAttribute || undefined,
  };

  const {
    data: timelines,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useTimeline(queryParams);

  const lanes = timelines ?? [];

  const laneBadgeClass =
    'px-2 py-0.5 rounded-full text-xs font-semibold bg-slate-800 ' +
    'text-slate-300 border border-slate-700';

  const filterBoxClass =
    'flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5';

  const refreshBtnClass =
    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 ' +
    'hover:bg-slate-700 text-slate-200 text-xs font-medium border ' +
    'border-slate-700 transition-colors cursor-pointer disabled:opacity-50';

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div
        className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4"
      >
        <div>
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">Slot Timeline</h2>
            {lanes.length > 0 && (
              <span className={laneBadgeClass}>
                {lanes.length} {lanes.length === 1 ? 'lane' : 'lanes'}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Temporal visualization of memory lifecycles, validity intervals, and revisions.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Entity Filter */}
          <div className={filterBoxClass}>
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedEntity}
              onChange={(e) => setSelectedEntity(e.target.value)}
              className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="">All Entities</option>
              {catalog?.entities?.map((ent, idx) => {
                const name =
                  typeof ent.name === 'string'
                    ? ent.name
                    : String(ent.name ?? `entity-${idx}`);
                return (
                  <option key={name} value={name}>
                    {name}
                  </option>
                );
              })}
            </select>
          </div>

          {/* Attribute Filter */}
          <div className={filterBoxClass}>
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={selectedAttribute}
              onChange={(e) => setSelectedAttribute(e.target.value)}
              className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="">All Attributes</option>
              {catalog?.attributes?.map((attr) => (
                <option key={attr.attribute} value={attr.attribute}>
                  {attr.attribute}
                </option>
              ))}
            </select>
          </div>

          {/* Refresh Button */}
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className={refreshBtnClass}
            title="Refresh timeline"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Loading & Error States */}
      {isLoading && <LoadingSpinner label="Loading slot timelines..." />}

      {isError && (
        <ErrorState
          error={error}
          title="Failed to load timeline"
          onRetry={() => refetch()}
        />
      )}

      {/* Empty State */}
      {!isLoading && !isError && lanes.length === 0 && (
        <EmptyState
          icon={Clock}
          title="No Timeline Lanes"
          description="Ingest messages to see temporal slot lanes and historical memory revisions."
        />
      )}

      {/* Timeline Lanes List */}
      {!isLoading && !isError && lanes.length > 0 && (
        <div className="space-y-6">
          {lanes.map((lane) => {
            const laneKey = getTimelineLaneKey(lane);
            return (
              <TimelineLaneCard
                key={laneKey}
                lane={lane}
                onSelectMemory={(id) => setInspectMemoryId(id)}
              />
            );
          })}
        </div>
      )}

      {/* Provenance Detail Drawer */}
      <ProvenanceDrawer
        memoryId={inspectMemoryId}
        onClose={() => setInspectMemoryId(null)}
      />
    </div>
  );
}
