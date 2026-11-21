import React, { useState, useEffect, useMemo } from 'react';
import {
  UploadCloud,
  Send,
  Sparkles,
  Calendar,
  AlertCircle,
  Copy,
  Layers,
  Play,
  Square,
  CheckCircle2,
} from 'lucide-react';
import { useIngest, useCatalog, useResetDemo } from '../api/hooks';
import type { IngestReport } from '../api/client';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { DecisionCard } from '../components/common/DecisionCard';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';
import { validateEventTimestamp } from '../lib/validation';
import { startReplay, type ReplaySession } from '../lib/replay';

export function IngestPage(): React.JSX.Element {
  const {
    data: catalog,
    isLoading: catalogLoading,
    error: catalogError,
    refetch: refetchCatalog,
  } = useCatalog();
  const ingestMutation = useIngest();
  const resetDemoMutation = useResetDemo();

  // Form state - sourceId initialized to empty string (not hardcoded)
  const [sourceId, setSourceId] = useState('');
  const [eventTime, setEventTime] = useState('');
  const [mode, setMode] = useState<'e2e' | 'structured'>('e2e');
  const [text, setText] = useState('');
  const [structuredClaimsJson, setStructuredClaimsJson] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);

  // Response & inspection state
  const [result, setResult] = useState<IngestReport | null>(null);
  const [selectedMemoryId, setSelectedMemoryId] = useState<number | null>(null);

  // SSE Scenario Replay state
  const [isReplaying, setIsReplaying] = useState(false);
  const [replaySession, setReplaySession] = useState<ReplaySession | null>(null);
  const [replayReports, setReplayReports] = useState<IngestReport[]>([]);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [replayCompleted, setReplayCompleted] = useState(false);
  const [replayMode, setReplayMode] = useState<'structured' | 'e2e'>('structured');
  const [showResetConfirmation, setShowResetConfirmation] = useState(false);

  const availableSources = useMemo(() => catalog?.sources ?? [], [catalog?.sources]);
  const hasValidSources = availableSources.length > 0;
  const isSourceValid = hasValidSources && availableSources.some((s) => s.id === sourceId);

  // Synchronize sourceId with catalog sources once loaded if currently empty or invalid
  useEffect(() => {
    if (hasValidSources) {
      if (!sourceId || !availableSources.some((s) => s.id === sourceId)) {
        setSourceId(availableSources[0].id);
      }
    }
  }, [hasValidSources, availableSources, sourceId]);

  // Preset hero scenarios for one-click testing
  const presets = [
    {
      label: '1. Initial State (Python 3.10)',
      source: 'sysadmin_priya',
      text: 'Server 1 was deployed running Python 3.10.',
      time: '2026-06-01T10:00:00Z',
    },
    {
      label: '2. Corroboration (3.10 check)',
      source: 'crawler_bot',
      text: 'Automated audit: Server 1 confirmed running Python 3.10.',
      time: '2026-06-02T10:00:00Z',
    },
    {
      label: '3. Update (Python 3.11)',
      source: 'devops_lead_marcus',
      text: 'Server 1 upgraded to Python 3.11 today.',
      time: '2026-06-03T10:00:00Z',
    },
    {
      label: '4. Conflict (Disputed 3.9)',
      source: 'intern_alex',
      text: 'Server 1 downgraded to Python 3.9.',
      time: '2026-06-04T10:00:00Z',
    },
    {
      label: '5. Zero Claims (Chatter)',
      source: 'intern_alex',
      text: 'Good morning everyone! Let me know if there are any updates.',
      time: '2026-06-05T09:00:00Z',
    },
  ];

  // Clean up any ongoing SSE connection on unmount or session replace
  useEffect(() => {
    return () => {
      replaySession?.stop();
    };
  }, [replaySession]);

  const handleExecuteReplay = () => {
    if (isReplaying || ingestMutation.isPending) {
      return;
    }
    setResult(null);
    setReplayReports([]);
    setReplayError(null);
    setReplayCompleted(false);
    setIsReplaying(true);

    const session = startReplay({
      scenario: 'hero-001',
      mode: replayMode,
      delay_ms: 700,
      onMessageIngested: (report) => {
        setReplayReports((prev) => [...prev, report]);
      },
      onDone: () => {
        setIsReplaying(false);
        setReplayCompleted(true);
        setReplaySession(null);
      },
      onError: (err) => {
        setIsReplaying(false);
        setReplayError(err instanceof Error ? err.message : err.error);
        setReplaySession(null);
      },
    });

    setReplaySession(session);
  };

  const handleReplayWithoutReset = () => {
    setShowResetConfirmation(false);
    handleExecuteReplay();
  };

  const handleConfirmAndResetReplay = async () => {
    setShowResetConfirmation(false);
    if (isReplaying || ingestMutation.isPending || resetDemoMutation.isPending) {
      return;
    }
    setResult(null);
    setReplayReports([]);
    setReplayError(null);
    setReplayCompleted(false);

    try {
      await resetDemoMutation.mutateAsync();
      handleExecuteReplay();
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Failed to reset database before replay.';
      setReplayError(message);
    }
  };

  const handleCancelReplay = () => {
    setShowResetConfirmation(false);
  };

  const handleStopReplay = () => {
    replaySession?.stop();
    setIsReplaying(false);
    setReplaySession(null);
  };

  const handleApplyPreset = (preset: typeof presets[0]) => {
    if (isReplaying) return;
    // Only select source IDs that exist in the loaded catalog
    let targetSource = '';
    if (availableSources.some((s) => s.id === preset.source)) {
      targetSource = preset.source;
    } else if (hasValidSources) {
      targetSource = availableSources[0].id;
    }
    setSourceId(targetSource);
    setText(preset.text);
    setEventTime(preset.time);
    setMode('e2e');
    setValidationError(null);
  };

  const handleSetTimeToNow = () => {
    setEventTime(new Date().toISOString().slice(0, 19) + 'Z');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isReplaying) return;
    setValidationError(null);

    // Guard: catalog must be loaded and have valid sources
    if (catalogLoading) {
      setValidationError('Please wait: catalog sources are still loading.');
      return;
    }
    if (catalogError) {
      setValidationError(
        'Cannot submit ingest message: catalog sources could not be loaded.'
      );
      return;
    }
    if (!hasValidSources) {
      setValidationError('Cannot submit ingest message: no catalog sources available.');
      return;
    }
    if (!sourceId || !isSourceValid) {
      setValidationError('Please select a valid source from the catalog.');
      return;
    }

    const trimmedText = text.trim();
    if (!trimmedText) {
      setValidationError('Message text cannot be empty.');
      return;
    }

    // Validate event_time format and calendar date
    const validation = validateEventTimestamp(eventTime);
    if (!validation.isValid) {
      setValidationError(validation.error!);
      return;
    }
    const isoEventTime = validation.value!;

    let parsedClaims = undefined;
    if (mode === 'structured') {
      if (!structuredClaimsJson.trim()) {
        setValidationError('Structured claims JSON is required when Structured mode is active.');
        return;
      }
      try {
        parsedClaims = JSON.parse(structuredClaimsJson);
        if (!Array.isArray(parsedClaims)) {
          setValidationError('Structured claims must be a JSON array of claim objects.');
          return;
        }
      } catch (err) {
        setValidationError(
          `Invalid JSON in structured claims: ${err instanceof Error ? err.message : String(err)}`
        );
        return;
      }
    }

    // Clear previous result and replay state before starting manual ingest
    setResult(null);
    setReplayReports([]);
    setReplayError(null);
    setReplayCompleted(false);

    try {
      const response = await ingestMutation.mutateAsync({
        source_id: sourceId,
        text: trimmedText,
        event_time: isoEventTime,
        mode,
        claims: parsedClaims,
      });
      setResult(response);
    } catch {
      // Error handled by mutation state
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Page Header */}
      <div
        className="border-b border-slate-800 pb-4 flex flex-col sm:flex-row
                   sm:items-center justify-between gap-2"
      >
        <div>
          <h2 className="text-lg font-bold text-white tracking-tight flex items-center space-x-2">
            <UploadCloud className="w-5 h-5 text-indigo-400" />
            <span>Ingest Messages</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Submit raw natural language or structured claims for entity resolution and
            conflict-aware resolution rules.
          </p>
        </div>

        {/* Hero Scenario Replay and Presets Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Replay Hero Scenario Button */}
          <div
            className="flex items-center space-x-1 p-0.5 rounded-lg bg-slate-900
                       border border-slate-800"
          >
            <button
              type="button"
              onClick={
                isReplaying
                  ? handleStopReplay
                  : () => setShowResetConfirmation(true)
              }
              disabled={
                !isReplaying &&
                (ingestMutation.isPending || catalogLoading || resetDemoMutation.isPending)
              }
              className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center
                         space-x-1.5 transition ${
                           isReplaying
                             ? 'bg-rose-600 hover:bg-rose-500 text-white'
                             : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm'
                         }`}
            >
              {isReplaying ? (
                <>
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Stop ({replayReports.length}/8)</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Replay Hero Scenario</span>
                </>
              )}
            </button>
            <select
              value={replayMode}
              onChange={(e) => setReplayMode(e.target.value as 'structured' | 'e2e')}
              disabled={isReplaying}
              className="bg-transparent text-[11px] font-mono text-slate-300 px-2 py-1
                         focus:outline-none"
              title="Select scenario replay mode"
            >
              <option value="structured" className="bg-slate-900 text-slate-200">
                Structured
              </option>
              <option value="e2e" className="bg-slate-900 text-slate-200">
                E2E (Natural)
              </option>
            </select>
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span
              className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider
                         flex items-center space-x-1"
            >
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>Presets:</span>
            </span>
            {presets.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => handleApplyPreset(p)}
                disabled={isReplaying}
                className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800
                           disabled:opacity-50 border border-slate-800 text-xs font-mono
                           text-indigo-300 transition"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Ingest Submission Form */}
        <div className="lg:col-span-6 space-y-4">
          <form
            onSubmit={handleSubmit}
            className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4"
          >
            {/* Mode Toggle */}
            <div
              className="flex items-center space-x-2 p-1 rounded-lg bg-slate-950/80
                         border border-slate-800 text-xs"
            >
              <button
                type="button"
                onClick={() => setMode('e2e')}
                className={`flex-1 py-1.5 rounded-md font-medium transition ${
                  mode === 'e2e'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                End-to-End (Natural Language)
              </button>
              <button
                type="button"
                onClick={() => setMode('structured')}
                className={`flex-1 py-1.5 rounded-md font-medium transition ${
                  mode === 'structured'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Structured Claims (Bypass LLM)
              </button>
            </div>

            {/* Source and Event Time Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-slate-400 font-medium mb-1">
                  Source ID *
                </label>
                {catalogLoading ? (
                  <LoadingSpinner size="sm" />
                ) : catalogError ? (
                  <div
                    className="p-2.5 rounded-lg border border-rose-900/60 bg-rose-950/40
                               text-xs space-y-1.5"
                  >
                    <div className="flex items-center space-x-1.5 text-rose-300">
                      <AlertCircle className="w-3.5 h-3.5 shrink-0 text-rose-400" />
                      <span className="font-semibold">Failed to load sources</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => refetchCatalog()}
                      className="px-2 py-0.5 rounded bg-rose-900/60 hover:bg-rose-800/80
                                 text-rose-200 text-[11px] font-medium transition"
                    >
                      Retry
                    </button>
                  </div>
                ) : !hasValidSources ? (
                  <div
                    className="p-2.5 rounded-lg border border-amber-900/60 bg-amber-950/40
                               text-xs text-amber-300"
                  >
                    No sources configured in catalog.
                  </div>
                ) : (
                  <select
                    value={sourceId}
                    onChange={(e) => setSourceId(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800
                               text-slate-200 text-xs focus:outline-none focus:border-indigo-500
                               font-mono"
                  >
                    {!sourceId && (
                      <option value="" disabled>
                        Select a source...
                      </option>
                    )}
                    {availableSources.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name} ({s.id}) &bull; trust: {s.reliability}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-slate-400 font-medium">
                    Event Time (UTC) *
                  </label>
                  <button
                    type="button"
                    onClick={handleSetTimeToNow}
                    className="text-[10px] text-indigo-400 hover:underline flex items-center
                               space-x-1"
                  >
                    <Calendar className="w-3 h-3" />
                    <span>Now</span>
                  </button>
                </div>
                <input
                  type="text"
                  value={eventTime}
                  onChange={(e) => setEventTime(e.target.value)}
                  placeholder="2026-06-01T10:00:00Z"
                  className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800
                             text-slate-200 text-xs focus:outline-none focus:border-indigo-500
                             font-mono"
                />
              </div>
            </div>

            {/* Message Text Area */}
            <div className="text-xs">
              <label className="block text-slate-400 font-medium mb-1">
                Raw Message Text *
              </label>
              <textarea
                rows={4}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter information to remember (e.g. Server 1 on Python 3.12)..."
                className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800
                           text-slate-200 text-xs focus:outline-none focus:border-indigo-500
                           font-sans leading-relaxed resize-none"
              />
            </div>

            {/* Optional Structured Claims JSON */}
            {mode === 'structured' && (
              <div className="text-xs space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-slate-400 font-medium">
                    Structured Claims (JSON array) *
                  </label>
                  <button
                    type="button"
                    onClick={() =>
                      setStructuredClaimsJson(
                        JSON.stringify(
                          [
                            {
                              subject: 'Server 1',
                              attribute: 'python_version',
                              value: '3.12',
                              polarity: 'assert',
                              extract_confidence: 0.95,
                            },
                          ],
                          null,
                          2
                        )
                      )
                    }
                    className="text-[10px] text-indigo-400 hover:underline"
                  >
                    Insert Example
                  </button>
                </div>
                <textarea
                  rows={5}
                  value={structuredClaimsJson}
                  onChange={(e) => setStructuredClaimsJson(e.target.value)}
                  placeholder='[{"subject": "Server 1", "attribute": "python_version", ...}]'
                  className="w-full p-2.5 rounded-lg bg-slate-950 border border-slate-800
                             text-slate-300 font-mono text-[11px] focus:outline-none
                             focus:border-indigo-500"
                />
              </div>
            )}

            {/* Client-side Validation Error */}
            {validationError && (
              <div
                className="p-3 rounded-lg border border-rose-900/60 bg-rose-950/40
                           text-rose-300 text-xs flex items-center space-x-2"
              >
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{validationError}</span>
              </div>
            )}

            {/* Server Error State */}
            {ingestMutation.isError && (
              <ErrorState
                error={ingestMutation.error}
                title="Ingestion Failed"
              />
            )}

            {/* Submit Button - disabled if no valid catalog source or replay running */}
            <button
              type="submit"
              disabled={
                ingestMutation.isPending ||
                catalogLoading ||
                Boolean(catalogError) ||
                !isSourceValid ||
                isReplaying
              }
              className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg
                         bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white
                         text-xs font-semibold shadow-sm transition"
            >
              {ingestMutation.isPending ? (
                <LoadingSpinner size="sm" label="Resolving & Ingesting..." />
              ) : isReplaying ? (
                <span>Replay in Progress (Inputs Disabled)</span>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Submit Ingest Message</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Ingest Result Output Area */}
        <div className="lg:col-span-6 space-y-4">
          <div
            className="p-5 rounded-xl border border-slate-800 bg-slate-900/30
                       space-y-4 min-h-[380px]"
          >
            {/* Header: Replay Mode vs Single Ingestion Mode */}
            {isReplaying || replayReports.length > 0 || replayError || replayCompleted ? (
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <div className="flex items-center space-x-2">
                  <Play className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">
                    Hero Scenario Replay Stream
                  </h3>
                </div>
                <span
                  className={`text-[11px] font-mono px-2 py-0.5 rounded border ${
                    isReplaying
                      ? 'border-indigo-800/80 bg-indigo-950/40 text-indigo-300'
                      : replayCompleted
                      ? 'border-emerald-800/80 bg-emerald-950/40 text-emerald-300'
                      : 'border-slate-800 bg-slate-900 text-slate-400'
                  }`}
                >
                  {isReplaying
                    ? `Streaming (${replayReports.length}/8)`
                    : replayCompleted
                    ? 'Completed (8/8)'
                    : 'Stopped'}
                </span>
              </div>
            ) : (
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <div className="flex items-center space-x-2">
                  <Layers className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">
                    Ingestion Decision Report
                  </h3>
                </div>
                {result && (
                  <span className="text-[11px] font-mono text-slate-400">
                    Message #{result.message_id}
                  </span>
                )}
              </div>
            )}

            {/* Replay Stream View */}
            {isReplaying || replayReports.length > 0 || replayError || replayCompleted ? (
              <div className="space-y-4">
                {/* Replay Error Alert */}
                {replayError && (
                  <div
                    className="p-4 rounded-xl border border-rose-900/60 bg-rose-950/30
                               text-rose-200 text-xs space-y-1.5"
                  >
                    <div className="flex items-center space-x-1.5 font-semibold text-rose-300">
                      <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                      <span>Replay Error</span>
                    </div>
                    <p className="text-slate-400">{replayError}</p>
                  </div>
                )}

                {/* Replay Summary Status Banner */}
                {replayCompleted ? (
                  <div
                    className="p-4 rounded-xl border border-emerald-800/80 bg-emerald-950/30
                               text-emerald-200 text-xs space-y-1"
                  >
                    <div className="flex items-center space-x-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span className="font-bold text-sm">Scenario Replay Finished</span>
                    </div>
                    <p className="text-slate-400 text-xs">
                      Successfully processed all 8 canonical hero messages sequentially.
                      Database was reset and verified.
                    </p>
                  </div>
                ) : isReplaying ? (
                  <div
                    className="p-4 rounded-xl border border-indigo-800/80 bg-indigo-950/30
                               text-indigo-200 text-xs space-y-1"
                  >
                    <div className="flex items-center space-x-2">
                      <LoadingSpinner size="sm" />
                      <span className="font-bold text-sm">Streaming Decisions in Real Time</span>
                    </div>
                    <p className="text-slate-400 text-xs">
                      Messages are ingested sequentially via SSE. Each message updates the
                      graph and resolves conflicts according to canonical engine rules.
                    </p>
                  </div>
                ) : null}

                {/* Accumulated Decision Cards in Sequential Order */}
                {replayReports.length > 0 ? (
                  <div className="space-y-4">
                    {replayReports.map((report) => (
                      <div key={report.message_id} className="space-y-2">
                        <div
                          className="flex items-center justify-between text-[11px] font-mono
                                     text-slate-400 border-b border-slate-800/60 pb-1"
                        >
                          <span>Message #{report.message_id}</span>
                          <span>
                            {report.decisions?.length ?? 0} decision(s) &bull;{' '}
                            {report.claims_extracted} claim(s)
                          </span>
                        </div>
                        <div className="space-y-2.5">
                          {report.decisions?.map((decision) => (
                            <DecisionCard
                              key={decision.id}
                              decision={decision}
                              onMemoryClick={(id) => setSelectedMemoryId(id)}
                            />
                          ))}
                        </div>
                      </div>
                    ))}

                    {isReplaying && (
                      <div className="py-4 flex justify-center text-xs text-slate-500">
                        <LoadingSpinner size="sm" label="Awaiting next message report..." />
                      </div>
                    )}
                  </div>
                ) : isReplaying ? (
                  <div className="py-16 text-center">
                    <LoadingSpinner
                      size="lg"
                      label="Resetting database and starting hero scenario replay..."
                    />
                  </div>
                ) : null}
              </div>
            ) : ingestMutation.isPending ? (
              <div
                className="py-20 flex flex-col items-center justify-center
                           space-x-2 text-slate-400"
              >
                <LoadingSpinner
                  size="lg"
                  label="Extracting claims and executing resolution rules..."
                />
              </div>
            ) : ingestMutation.isError ? (
              <div
                className="p-4 rounded-xl border border-rose-900/60 bg-rose-950/30
                           text-rose-200 text-xs space-y-1.5"
              >
                <div className="flex items-center space-x-1.5 font-semibold text-rose-300">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                  <span>Ingestion Failed</span>
                </div>
                <p className="text-slate-400">
                  {ingestMutation.error instanceof Error
                    ? ingestMutation.error.message
                    : 'The ingestion request could not be processed.'}
                </p>
              </div>
            ) : result ? (
              <div className="space-y-4">
                {/* Result Summary Banner */}
                <div
                  className={`p-4 rounded-xl border text-xs space-y-2 ${
                    result.duplicate
                      ? 'border-amber-800/80 bg-amber-950/30 text-amber-200'
                      : result.claims_extracted === 0
                      ? 'border-slate-800 bg-slate-950/50 text-slate-400'
                      : 'border-emerald-800/80 bg-emerald-950/30 text-emerald-200'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm">
                      {result.duplicate
                        ? 'Duplicate Message Detected'
                        : result.claims_extracted === 0
                        ? 'Zero Claims Extracted'
                        : 'Ingest & Resolution Completed'}
                    </span>
                    <span
                      className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-900/80
                                 border border-slate-800"
                    >
                      claims: {result.claims_extracted}
                    </span>
                  </div>

                  <p className="text-xs leading-relaxed opacity-90">
                    {result.duplicate
                      ? 'This exact message payload was previously ingested from this source. ' +
                        'Idempotency rule skipped re-execution.'
                      : result.claims_extracted === 0
                      ? 'No verifiable factual claims matching catalog entities or attributes ' +
                        'were extracted from this message.'
                      : `Successfully processed ${result.claims_extracted} claim(s) and logged ` +
                        `${result.decisions?.length ?? 0} resolution decision(s).`}
                  </p>
                </div>

                {/* Decision Cards List */}
                {result.decisions && result.decisions.length > 0 && (
                  <div className="space-y-3">
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                      Resolution Actions ({result.decisions.length})
                    </p>
                    <div className="space-y-2.5">
                      {result.decisions.map((decision) => (
                        <DecisionCard
                          key={decision.id}
                          decision={decision}
                          onMemoryClick={(id) => setSelectedMemoryId(id)}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-20 text-center text-xs text-slate-500 space-y-2">
                <Copy className="w-8 h-8 text-slate-700 mx-auto" />
                <p>
                  Submit an ingest message or click Replay Hero Scenario to view resolution
                  decisions.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Slide-over Provenance Drawer */}
      <ProvenanceDrawer
        memoryId={selectedMemoryId}
        onClose={() => setSelectedMemoryId(null)}
        onSelectMemory={(id) => setSelectedMemoryId(id)}
      />

      {/* Destructive Reset Confirmation Dialog */}
      {showResetConfirmation && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="reset-modal-title"
          className="fixed inset-0 z-50 flex items-center justify-center
                     bg-black/75 backdrop-blur-xs p-4"
        >
          <div
            className="w-full max-w-md rounded-xl border border-rose-900/60
                       bg-slate-900 p-6 shadow-2xl space-y-4"
          >
            <div className="flex items-center space-x-3 text-rose-400">
              <AlertCircle className="w-6 h-6 shrink-0" />
              <h3 id="reset-modal-title" className="text-sm font-bold text-white">
                Confirm Database Reset &amp; Replay
              </h3>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Replaying with database reset will{' '}
              <strong className="text-rose-400 font-semibold">
                permanently delete ALL
              </strong>{' '}
              existing MemoryMesh application data (including all memories, claims,
              disputes, and resolution decisions), not just previous demo records.
            </p>

            <p className="text-xs text-slate-400">
              You can choose to reset and start clean, replay without resetting, or cancel.
            </p>

            <div className="flex flex-col sm:flex-row gap-2 pt-2 justify-end">
              <button
                type="button"
                onClick={handleCancelReplay}
                disabled={resetDemoMutation.isPending}
                className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800
                           hover:bg-slate-700 disabled:opacity-50 text-xs text-slate-300
                           font-medium transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleReplayWithoutReset}
                disabled={resetDemoMutation.isPending}
                className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800
                           hover:bg-slate-700 disabled:opacity-50 text-xs text-indigo-300
                           font-medium transition"
              >
                Replay Without Reset
              </button>
              <button
                type="button"
                onClick={handleConfirmAndResetReplay}
                disabled={resetDemoMutation.isPending}
                className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500
                           disabled:opacity-50 text-xs text-white font-semibold
                           shadow-sm transition"
              >
                {resetDemoMutation.isPending ? 'Resetting...' : 'Confirm & Reset All'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
