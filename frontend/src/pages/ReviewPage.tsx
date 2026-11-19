import React, { useState } from 'react';
import {
  Scale,
  Clock,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  X,
} from 'lucide-react';
import {
  useDisputes,
  useMemories,
  useResolveDispute,
  useAcknowledgeMemory,
} from '../api/hooks';
import { computeReviewQueueCounts } from '../lib/review';
import { DisputeCard } from '../components/review/DisputeCard';
import { ReviewMemoryCard } from '../components/review/ReviewMemoryCard';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';

export function ReviewPage(): React.JSX.Element {
  const [inspectMemoryId, setInspectMemoryId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'disputes' | 'flagged'>('all');
  const [actionFeedback, setActionFeedback] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const {
    data: disputesData,
    isLoading: isLoadingDisputes,
    isError: isErrorDisputes,
    error: errorDisputes,
    refetch: refetchDisputes,
    isFetching: isFetchingDisputes,
  } = useDisputes();

  const {
    data: flaggedData,
    isLoading: isLoadingFlagged,
    isError: isErrorFlagged,
    error: errorFlagged,
    refetch: refetchFlagged,
    isFetching: isFetchingFlagged,
  } = useMemories({ needs_review: true });

  const resolveMutation = useResolveDispute();
  const acknowledgeMutation = useAcknowledgeMemory();

  const disputes = disputesData ?? [];
  const flaggedMemories = flaggedData ?? [];
  const { totalPending } = computeReviewQueueCounts(disputes, flaggedMemories);
  const isRefreshing = isFetchingDisputes || isFetchingFlagged;
  const isLoading = isLoadingDisputes || isLoadingFlagged;
  const isError = isErrorDisputes || isErrorFlagged;
  const generalError = errorDisputes || errorFlagged;

  const handleResolveDispute = async (winnerMemoryId: number) => {
    setActionFeedback(null);
    try {
      const decision = await resolveMutation.mutateAsync({
        winner_memory_id: winnerMemoryId,
      });
      setActionFeedback({
        type: 'success',
        message:
          `Dispute resolved. Memory #${winnerMemoryId} ` +
          `confirmed as winner via ${decision.rule_id}.`,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to resolve dispute.';
      setActionFeedback({
        type: 'error',
        message: msg,
      });
    }
  };

  const handleAcknowledgeMemory = async (memoryId: number) => {
    setActionFeedback(null);
    try {
      await acknowledgeMutation.mutateAsync(memoryId);
      setActionFeedback({
        type: 'success',
        message: `Review acknowledged for Memory #${memoryId}. Flag cleared successfully.`,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to acknowledge memory review.';
      setActionFeedback({
        type: 'error',
        message: msg,
      });
    }
  };

  const handleRefreshAll = () => {
    setActionFeedback(null);
    refetchDisputes();
    refetchFlagged();
  };

  const pendingBadgeClass =
    'px-2 py-0.5 rounded-full text-xs font-semibold bg-amber-950/80 ' +
    'text-amber-300 border border-amber-800/80';

  const tabGroupClass =
    'flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs';

  const closeBtnClass =
    'p-1 hover:bg-slate-800/60 rounded text-slate-400 hover:text-white cursor-pointer';

  const refreshBtnClass =
    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 ' +
    'hover:bg-slate-700 text-slate-200 text-xs font-medium border ' +
    'border-slate-700 transition-colors cursor-pointer disabled:opacity-50';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div
        className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4"
      >
        <div>
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              Disputes & Review
            </h2>
            {totalPending > 0 && (
              <span className={pendingBadgeClass}>
                {totalPending} pending
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Review flagged memories, inspect conflicting claims, select winners, and acknowledge.
          </p>
        </div>

        {/* Tab Selector & Refresh */}
        <div className="flex flex-wrap items-center gap-2">
          <div className={tabGroupClass}>
            <button
              type="button"
              onClick={() => setActiveTab('all')}
              className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                activeTab === 'all'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              All ({totalPending})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('disputes')}
              className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                activeTab === 'disputes'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Disputes ({disputes.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('flagged')}
              className={`px-3 py-1 rounded-md font-medium transition cursor-pointer ${
                activeTab === 'flagged'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Flagged ({flaggedMemories.length})
            </button>
          </div>

          <button
            type="button"
            onClick={handleRefreshAll}
            disabled={isRefreshing}
            className={refreshBtnClass}
            title="Refresh review queue"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Action Notification Banner */}
      {actionFeedback && (
        <div
          className={`p-3.5 rounded-xl border flex items-center justify-between gap-3 text-xs ${
            actionFeedback.type === 'success'
              ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200'
              : 'bg-rose-950/40 border-rose-800/60 text-rose-200'
          }`}
        >
          <div className="flex items-center gap-2">
            {actionFeedback.type === 'success' ? (
              <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            )}
            <span>{actionFeedback.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setActionFeedback(null)}
            className={closeBtnClass}
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Loading & Error States */}
      {isLoading && <LoadingSpinner label="Loading review items and disputes..." />}

      {isError && (
        <ErrorState
          error={generalError}
          title="Failed to load review items"
          onRetry={handleRefreshAll}
        />
      )}

      {/* Empty State */}
      {!isLoading && !isError && totalPending === 0 && (
        <EmptyState
          icon={CheckCircle}
          title="All Caught Up"
          description="There are currently no active disputes or flagged memories requiring review."
        />
      )}

      {/* Content Sections */}
      {!isLoading && !isError && totalPending > 0 && (
        <div className="space-y-8">
          {/* Section 1: Active Disputes */}
          {(activeTab === 'all' || activeTab === 'disputes') && disputes.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800/60 pb-2">
                <Scale className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                  Active Disputes ({disputes.length})
                </h3>
              </div>

              <div className="space-y-4">
                {disputes.map((dispute, idx) => {
                  const key = `dispute-${dispute.a.memory.id}-${dispute.b.memory.id}-${idx}`;
                  return (
                    <DisputeCard
                      key={key}
                      dispute={dispute}
                      onSelectMemory={(id) => setInspectMemoryId(id)}
                      onResolve={handleResolveDispute}
                      isResolving={resolveMutation.isPending}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Section 2: Flagged Memories */}
          {(activeTab === 'all' || activeTab === 'flagged') && flaggedMemories.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 border-b border-slate-800/60 pb-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                  Memories Awaiting Review ({flaggedMemories.length})
                </h3>
              </div>

              <div className="space-y-4">
                {flaggedMemories.map((mv) => (
                  <ReviewMemoryCard
                    key={mv.memory.id}
                    memoryView={mv}
                    onSelectMemory={(id) => setInspectMemoryId(id)}
                    onAcknowledge={handleAcknowledgeMemory}
                    isAcknowledging={acknowledgeMutation.isPending}
                  />
                ))}
              </div>
            </div>
          )}
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
