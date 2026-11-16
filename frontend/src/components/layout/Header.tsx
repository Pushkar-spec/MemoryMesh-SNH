import React, { useState } from 'react';
import { Layers, RotateCcw, Check, Server } from 'lucide-react';
import { useHealth, useResetDemo } from '../../api/hooks';
import { LoadingSpinner } from '../common/LoadingSpinner';

export function Header(): React.JSX.Element {
  const { data: health, isLoading: healthLoading, isError: healthError } = useHealth();
  const resetMutation = useResetDemo();
  const [resetSuccess, setResetSuccess] = useState(false);

  // Demo Clock display
  const [currentTime, setCurrentTime] = useState(() =>
    new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
  );

  React.useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(
        new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
      );
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleReset = async () => {
    if (resetMutation.isPending) return;
    try {
      await resetMutation.mutateAsync();
      setResetSuccess(true);
      setTimeout(() => setResetSuccess(false), 2500);
    } catch {
      // Error handled by mutation state
    }
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/70 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Brand */}
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-indigo-600 rounded-lg text-white shadow-sm shadow-indigo-500/30">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold text-white tracking-tight">MemoryMesh</h1>
            <span className="px-1.5 py-0.2 rounded text-[10px] font-mono font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800/80">
              v0.1
            </span>
          </div>
          <p className="text-xs text-slate-400">Conflict-Aware Long-Term AI Memory</p>
        </div>
      </div>

      {/* Center / Right Toolbar */}
      <div className="flex items-center space-x-4">
        {/* Demo Clock */}
        <div className="hidden md:flex items-center space-x-2 px-3 py-1 rounded-lg bg-slate-950/60 border border-slate-800 text-xs font-mono text-slate-300">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-slate-400">Demo Clock:</span>
          <span className="font-semibold text-slate-200">{currentTime}</span>
        </div>

        {/* Provider Badge from /api/health */}
        <div className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
          <Server className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400">LLM:</span>
          {healthLoading ? (
            <LoadingSpinner size="sm" />
          ) : healthError ? (
            <span className="inline-flex items-center text-rose-400 font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500 mr-1" />
              Offline
            </span>
          ) : (
            <span className="inline-flex items-center font-medium text-slate-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5" />
              <span className="capitalize">{health?.llm_provider || 'Connected'}</span>
            </span>
          )}
        </div>

        {/* Reset Demo Action */}
        <button
          type="button"
          onClick={handleReset}
          disabled={resetMutation.isPending}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
            resetSuccess
              ? 'bg-emerald-950 text-emerald-300 border-emerald-700'
              : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700 active:bg-slate-800'
          }`}
          title="Reset database and re-seed demo data"
        >
          {resetMutation.isPending ? (
            <LoadingSpinner size="sm" />
          ) : resetSuccess ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span>Reset Done</span>
            </>
          ) : (
            <>
              <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
              <span>Reset Demo</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
}
