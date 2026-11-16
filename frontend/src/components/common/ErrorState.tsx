import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import type { ApiError } from '../../api/client';

interface ErrorStateProps {
  error: Error | ApiError | string | null | undefined;
  title?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  error,
  title = 'Failed to load data',
  onRetry,
  className = '',
}: ErrorStateProps): React.JSX.Element {
  let message = 'An unexpected error occurred.';
  let code: string | null = null;
  let status: number | null = null;

  if (typeof error === 'string') {
    message = error;
  } else if (error && typeof error === 'object') {
    message = error.message || message;
    if ('code' in error && typeof error.code === 'string') {
      code = error.code;
    }
    if ('status' in error && typeof error.status === 'number' && error.status > 0) {
      status = error.status;
    }
  }

  return (
    <div
      className={`p-6 rounded-xl border border-rose-900/50 bg-rose-950/20 text-rose-200 ${className}`}
    >
      <div className="flex items-start space-x-3">
        <div className="p-2 rounded-lg bg-rose-900/40 text-rose-400 shrink-0">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <h4 className="text-sm font-semibold text-rose-200">{title}</h4>
            {code && (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-rose-900/60 text-rose-300 border border-rose-800">
                {code}
              </span>
            )}
            {status && (
              <span className="text-xs text-rose-400 font-mono">
                HTTP {status}
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-rose-300/90 leading-relaxed font-mono break-words">
            {message}
          </p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-900/60 hover:bg-rose-900/80 text-rose-200 text-xs font-medium border border-rose-800 transition"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
