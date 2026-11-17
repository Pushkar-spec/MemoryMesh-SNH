import React, { useState } from 'react';
import {
  MessageSquare,
  Search,
  Sparkles,
  Calendar,
  Split,
  HelpCircle,
  Clock,
} from 'lucide-react';
import { useAsk } from '../api/hooks';
import type { AskResponse } from '../api/client';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { AnswerCard } from '../components/ask/AnswerCard';
import { NaiveRAGCompareCard } from '../components/ask/NaiveRAGCompareCard';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';

export function AskPage(): React.JSX.Element {
  const askMutation = useAsk();

  // Query parameters state
  const [question, setQuestion] = useState('');
  const [temporalNow, setTemporalNow] = useState('');
  const [enableCompare, setEnableCompare] = useState(true);
  const [submittedQuestion, setSubmittedQuestion] = useState<string | null>(null);
  const [submittedNow, setSubmittedNow] = useState<string | null>(null);

  // Response & Drawer state
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [selectedMemoryId, setSelectedMemoryId] = useState<number | null>(null);

  // Example suggested questions
  const exampleQuestions = [
    {
      q: 'What Python version is Server 1 running?',
      desc: 'Hero scenario version query',
      now: '',
    },
    {
      q: 'What Python version was Server 1 running on June 1?',
      desc: 'Point-in-time temporal query',
      now: '2026-06-01T12:00:00Z',
    },
    {
      q: 'Which server is the primary database?',
      desc: 'Entity attribute lookup',
      now: '',
    },
    {
      q: 'What is the deployment status of Server 2?',
      desc: 'Status check',
      now: '',
    },
  ];

  const handleApplyExample = (ex: typeof exampleQuestions[0]) => {
    setQuestion(ex.q);
    setTemporalNow(ex.now);
  };

  const handleSetTimeToNow = () => {
    setTemporalNow(new Date().toISOString().slice(0, 19) + 'Z');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    let parsedNow = undefined;
    if (temporalNow.trim()) {
      parsedNow = temporalNow.trim();
    }

    setSubmittedQuestion(trimmed);
    setSubmittedNow(parsedNow || null);
    setResponse(null);

    try {
      const res = await askMutation.mutateAsync({
        question: trimmed,
        now: parsedNow,
        compare: enableCompare,
      });
      setResponse(res);
    } catch {
      // Error handled by mutation state
    }
  };

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center space-x-2">
          <MessageSquare className="w-5 h-5 text-indigo-400" />
          <span>Ask Long-Term Memory</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Query MemoryMesh with temporal reasoning, provenance citations, and side-by-side
          Naive RAG comparison.
        </p>
      </div>

      {/* Question Form */}
      <form
        onSubmit={handleSubmit}
        className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4"
      >
        <div className="relative">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about remembered systems, servers, and configurations..."
            className="w-full pl-10 pr-24 py-3 rounded-xl bg-slate-950 border border-slate-800
                       text-sm text-white focus:outline-none focus:border-indigo-500 font-sans
                       shadow-inner"
          />
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
          <button
            type="submit"
            disabled={askMutation.isPending || !question.trim()}
            className="absolute right-2 top-2 px-4 py-1.5 rounded-lg bg-indigo-600
                       hover:bg-indigo-500 disabled:opacity-50 text-white text-xs
                       font-semibold shadow-sm transition"
          >
            {askMutation.isPending ? <LoadingSpinner size="sm" /> : 'Ask'}
          </button>
        </div>

        {/* Query Controls: Temporal parameter & Naive RAG compare toggle */}
        <div
          className="flex flex-col sm:flex-row sm:items-center justify-between
                     gap-3 pt-1 text-xs"
        >
          {/* Temporal Parameter `now` */}
          <div className="flex items-center space-x-2">
            <Calendar className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            <span className="text-slate-400">Point-in-Time (`now`):</span>
            <input
              type="text"
              value={temporalNow}
              onChange={(e) => setTemporalNow(e.target.value)}
              placeholder="e.g. 2026-06-01T10:00:00Z"
              className="px-2.5 py-1 rounded bg-slate-950 border border-slate-800
                         text-slate-300 font-mono text-[11px] w-48 focus:outline-none
                         focus:border-indigo-500"
            />
            <button
              type="button"
              onClick={handleSetTimeToNow}
              className="text-[10px] text-indigo-400 hover:underline"
            >
              Current Time
            </button>
          </div>

          {/* Naive RAG Compare Toggle */}
          <label className="inline-flex items-center space-x-2 cursor-pointer text-slate-300">
            <input
              type="checkbox"
              checked={enableCompare}
              onChange={(e) => setEnableCompare(e.target.checked)}
              className="w-4 h-4 rounded bg-slate-950 border-slate-800 text-indigo-600
                         focus:ring-0 focus:ring-offset-0"
            />
            <span className="flex items-center space-x-1">
              <Split className="w-3.5 h-3.5 text-indigo-400" />
              <span>Compare with Naive RAG Baseline</span>
            </span>
          </label>
        </div>

        {/* Quick Example Suggestions */}
        <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-800/80">
          <span
            className="text-[11px] font-semibold text-slate-500 uppercase
                       tracking-wider flex items-center space-x-1"
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Try Examples:</span>
          </span>
          {exampleQuestions.map((ex) => (
            <button
              key={ex.q}
              type="button"
              onClick={() => handleApplyExample(ex)}
              className="px-2.5 py-1 rounded bg-slate-950 hover:bg-slate-800
                         border border-slate-800 text-xs text-slate-300 font-sans transition"
            >
              {ex.q}
            </button>
          ))}
        </div>
      </form>

      {/* Error State */}
      {askMutation.isError && (
        <ErrorState
          error={askMutation.error}
          title="Question Answering Failed"
        />
      )}

      {/* Answer & Comparison Results */}
      {askMutation.isPending ? (
        <div
          className="py-20 flex flex-col items-center justify-center space-y-3
                     text-slate-400"
        >
          <LoadingSpinner
            size="lg"
            label="Querying memory graph and resolving temporal state..."
          />
          <p className="text-xs text-slate-500">
            Extracting query slot &bull; Traversing edges &bull; Grounding factual citations
          </p>
        </div>
      ) : response ? (
        <div className="space-y-4">
          {/* Active Question Banner */}
          <div
            className="p-4 rounded-xl border border-slate-800 bg-slate-900/60
                       flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
          >
            <div className="space-y-0.5">
              <span
                className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]"
              >
                Queried Question:
              </span>
              <p className="text-sm font-semibold text-white">
                &ldquo;{submittedQuestion}&rdquo;
              </p>
            </div>
            {submittedNow && (
              <div
                className="flex items-center space-x-1.5 text-slate-400 font-mono text-[11px]
                           px-2.5 py-1 rounded bg-slate-950 border border-slate-800 shrink-0"
              >
                <Clock className="w-3.5 h-3.5 text-indigo-400" />
                <span>Reference Time: {submittedNow}</span>
              </div>
            )}
          </div>

          {/* Results Comparison Grid */}
          <div
            className={`grid gap-6 ${
              response.baseline ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'
            }`}
          >
            {/* MemoryMesh Answer Card */}
            <div>
              <AnswerCard
                result={response.mesh}
                onMemoryClick={(id) => setSelectedMemoryId(id)}
              />
            </div>

            {/* Naive RAG Baseline Comparison Card */}
            {response.baseline && (
              <div>
                <NaiveRAGCompareCard response={response} />
              </div>
            )}
          </div>
        </div>
      ) : !askMutation.isError ? (
        <div className="py-20 text-center text-xs text-slate-500 space-y-2">
          <HelpCircle className="w-8 h-8 text-slate-700 mx-auto" />
          <p>
            Ask a question or select an example above to query MemoryMesh long-term memory.
          </p>
        </div>
      ) : null}

      {/* Slide-over Provenance Drawer for Cited Memories */}
      <ProvenanceDrawer
        memoryId={selectedMemoryId}
        onClose={() => setSelectedMemoryId(null)}
        onSelectMemory={(id) => setSelectedMemoryId(id)}
      />
    </div>
  );
}
