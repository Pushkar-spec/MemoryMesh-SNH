import React from 'react';
import { Layers, Database, Activity, GitBranch, Terminal } from 'lucide-react';

export default function App(): React.JSX.Element {
  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
      {/* Top Navigation Shell */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-600 rounded-lg text-white">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">MemoryMesh</h1>
            <p className="text-xs text-slate-400">Conflict-Aware Long-Term AI Memory</p>
          </div>
        </div>

        <nav className="flex items-center space-x-1 text-sm font-medium text-slate-400">
          <span className="px-3 py-1.5 rounded-md text-slate-200 bg-slate-800 flex items-center space-x-1.5">
            <Database className="w-4 h-4" />
            <span>Memories</span>
          </span>
          <span className="px-3 py-1.5 rounded-md hover:text-slate-200 flex items-center space-x-1.5 cursor-not-allowed opacity-50">
            <GitBranch className="w-4 h-4" />
            <span>Graph</span>
          </span>
          <span className="px-3 py-1.5 rounded-md hover:text-slate-200 flex items-center space-x-1.5 cursor-not-allowed opacity-50">
            <Activity className="w-4 h-4" />
            <span>Decisions</span>
          </span>
          <span className="px-3 py-1.5 rounded-md hover:text-slate-200 flex items-center space-x-1.5 cursor-not-allowed opacity-50">
            <Terminal className="w-4 h-4" />
            <span>Inspector</span>
          </span>
        </nav>
      </header>

      {/* Main Content Area Placeholder Shell */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6">
        <div className="border border-dashed border-slate-800 rounded-xl p-12 text-center bg-slate-900/30">
          <div className="inline-flex p-3 bg-slate-800 rounded-full text-indigo-400 mb-4">
            <Layers className="w-8 h-8" />
          </div>
          <h2 className="text-xl font-semibold text-slate-200">MemoryMesh Scaffold Ready</h2>
          <p className="text-slate-400 max-w-md mx-auto mt-2 text-sm">
            Frontend foundation initialized with React 18, TypeScript, Vite, Tailwind CSS,
            TanStack Query, React Router, React Flow, and Lucide icons.
          </p>
          <div className="mt-6 flex justify-center gap-3 text-xs text-slate-500 font-mono">
            <span className="px-2.5 py-1 bg-slate-800/80 rounded border border-slate-700/50">
              Vite dev: 5173
            </span>
            <span className="px-2.5 py-1 bg-slate-800/80 rounded border border-slate-700/50">
              Proxy: /api &rarr; 8000
            </span>
            <span className="px-2.5 py-1 bg-slate-800/80 rounded border border-slate-700/50">
              React Flow: ready
            </span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 px-6 py-3 text-center text-xs text-slate-600">
        MemoryMesh &bull; P0.4 Scaffold &bull; API endpoints deferred to P9
      </footer>
    </div>
  );
}

