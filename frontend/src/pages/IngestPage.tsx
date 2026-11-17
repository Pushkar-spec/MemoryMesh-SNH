import React, { useState, useEffect, useMemo } from 'react';
import {
  UploadCloud,
  Send,
  Sparkles,
  Calendar,
  AlertCircle,
  Copy,
  Layers,
} from 'lucide-react';
import { useIngest, useCatalog } from '../api/hooks';
import type { IngestReport } from '../api/client';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { DecisionCard } from '../components/common/DecisionCard';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';
import { validateEventTimestamp } from '../lib/validation';

export function IngestPage(): React.JSX.Element {
  const {
    data: catalog,
    isLoading: catalogLoading,
    error: catalogError,
    refetch: refetchCatalog,
  } = useCatalog();
  const ingestMutation = useIngest();

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

  const handleApplyPreset = (preset: typeof presets[0]) => {
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

    // Clear previous result before starting request to avoid displaying stale report on error
    setResult(null);

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

        {/* Hero Scenario Quick Presets */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span
            className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider
                       flex items-center space-x-1"
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Hero Presets:</span>
          </span>
          {presets.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => handleApplyPreset(p)}
              className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800
                         border border-slate-800 text-xs font-mono text-indigo-300 transition"
            >
              {p.label}
            </button>
          ))}
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

            {/* Submit Button - disabled if no valid catalog source is selected */}
            <button
              type="submit"
              disabled={
                ingestMutation.isPending ||
                catalogLoading ||
                Boolean(catalogError) ||
                !isSourceValid
              }
              className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg
                         bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white
                         text-xs font-semibold shadow-sm transition"
            >
              {ingestMutation.isPending ? (
                <LoadingSpinner size="sm" label="Resolving & Ingesting..." />
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

            {ingestMutation.isPending ? (
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
                  Submit an ingest message or select a hero preset above to view resolution
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
    </div>
  );
}
