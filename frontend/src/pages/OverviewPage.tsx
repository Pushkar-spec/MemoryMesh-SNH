import React from 'react';
import { Layers, Database, ShieldCheck, Cpu, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useHealth, useCatalog } from '../api/hooks';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { StatusBadge } from '../components/common/StatusBadge';
import { ConfidenceBar } from '../components/common/ConfidenceBar';
import { ValidityBar } from '../components/common/ValidityBar';

export function OverviewPage(): React.JSX.Element {
  const { data: health, isLoading: healthLoading, error: healthError, refetch: refetchHealth } =
    useHealth();
  const { data: catalog, isLoading: catalogLoading, error: catalogError, refetch: refetchCatalog } =
    useCatalog();

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-950/80 text-indigo-300 border border-indigo-800/60 mb-2">
              <Layers className="w-3.5 h-3.5" />
              <span>P9.2 Frontend Shell & API Client Active</span>
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              MemoryMesh Dashboard
            </h2>
            <p className="text-xs text-slate-400 mt-1 max-w-xl">
              Conflict-aware long-term memory system for LLM applications. Centralized
              API client connected to FastAPI backend with TanStack Query and generated OpenAPI contracts.
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <Link
              to="/memories"
              className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition"
            >
              <span>Explore Memories</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Backend & Catalog Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: API & LLM Provider */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <span className="flex items-center space-x-1.5">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span>LLM Provider</span>
            </span>
            <span className="text-[10px] font-mono text-slate-500">GET /api/health</span>
          </div>

          {healthLoading ? (
            <LoadingSpinner size="sm" label="Connecting to backend..." />
          ) : healthError ? (
            <ErrorState error={healthError} onRetry={() => refetchHealth()} />
          ) : (
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Configured:</span>
                <span className="font-semibold text-slate-200 capitalize">
                  {health?.llm_provider}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Status:</span>
                <span className="inline-flex items-center text-emerald-400 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5" />
                  {health?.status}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Database:</span>
                <span className="font-mono text-slate-300">{health?.db}</span>
              </div>
            </div>
          )}
        </div>

        {/* Card 2: Configuration Catalog */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <span className="flex items-center space-x-1.5">
              <Database className="w-4 h-4 text-indigo-400" />
              <span>Catalog Schema</span>
            </span>
            <span className="text-[10px] font-mono text-slate-500">GET /api/catalog</span>
          </div>

          {catalogLoading ? (
            <LoadingSpinner size="sm" label="Loading catalog..." />
          ) : catalogError ? (
            <ErrorState error={catalogError} onRetry={() => refetchCatalog()} />
          ) : (
            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Sources:</span>
                <span className="font-mono font-semibold text-slate-200">
                  {catalog?.sources?.length ?? 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Attributes:</span>
                <span className="font-mono font-semibold text-slate-200">
                  {catalog?.attributes?.length ?? 0}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Entities:</span>
                <span className="font-mono font-semibold text-slate-200">
                  {catalog?.entities?.length ?? 0}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Card 3: Memory Rules & Verification */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <span className="flex items-center space-x-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Invariants & Rules</span>
            </span>
            <span className="text-[10px] font-mono text-slate-500">Engine V1</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Rules active:</span>
              <span className="font-mono font-semibold text-slate-200">
                R-INSERT, R-UPDATE, R-CORROBORATE, R-DISPUTE
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Conflict model:</span>
              <span className="font-medium text-slate-200">Deterministic + Disjoint</span>
            </div>
          </div>
        </div>
      </div>

      {/* Shared UI Components Showcase */}
      <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/20 space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            Shared Component Standards (P9.2)
          </h3>
          <p className="text-xs text-slate-400">
            Foundational presentation components configured for use across future feature pages.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
          {/* Status Badges */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Status Badges
            </p>
            <div className="flex flex-wrap gap-2">
              <StatusBadge status="active" />
              <StatusBadge status="historical" />
              <StatusBadge status="disputed" />
              <StatusBadge status="superseded" />
              <StatusBadge status="active" needsReview={true} />
            </div>
          </div>

          {/* Confidence Bars */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Confidence Bars
            </p>
            <div className="space-y-2">
              <ConfidenceBar score={0.95} />
              <ConfidenceBar score={0.65} />
              <ConfidenceBar score={0.32} />
            </div>
          </div>

          {/* Temporal Validity Bars */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Validity Intervals
            </p>
            <div className="space-y-2">
              <ValidityBar validFrom="2026-06-01T10:00:00Z" validTo={null} />
              <ValidityBar
                validFrom="2026-05-01T08:00:00Z"
                validTo="2026-05-31T23:59:59Z"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
