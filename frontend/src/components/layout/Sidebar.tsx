import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  UploadCloud,
  Database,
  Share2,
  Clock,
  MessageSquare,
  Scale,
  Activity,
  Terminal,
  BarChart3,
  Settings,
} from 'lucide-react';

interface NavItem {
  to: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  batch: string;
}

const navItems: NavItem[] = [
  { to: '/ingest', label: 'Ingest', icon: UploadCloud, batch: 'P9.3' },
  { to: '/memories', label: 'Memories', icon: Database, batch: 'P9.3' },
  { to: '/graph', label: 'Dependency Graph', icon: Share2, batch: 'P9.4' },
  { to: '/timeline', label: 'Timeline', icon: Clock, batch: 'P9.4' },
  { to: '/ask', label: 'Ask Memory', icon: MessageSquare, batch: 'P9.3' },
  { to: '/review', label: 'Disputes & Review', icon: Scale, batch: 'P9.5' },
  { to: '/decisions', label: 'Decision Log', icon: Activity, batch: 'P9.5' },
  { to: '/inspector', label: 'LLM Inspector', icon: Terminal, batch: 'P9.7' },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3, batch: 'P9.7' },
  { to: '/settings', label: 'Settings', icon: Settings, batch: 'P9.7' },
];

export function Sidebar(): React.JSX.Element {
  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-900/40 flex flex-col shrink-0">
      <div className="p-4 border-b border-slate-800/80">
        <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Navigation
        </p>
      </div>

      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/20'
                    : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'
                }`
              }
            >
              <div className="flex items-center space-x-2.5">
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </div>
              <span className="text-[10px] font-mono opacity-60 uppercase">
                {item.batch}
              </span>
            </NavLink>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800/80 text-[11px] text-slate-500">
        <div className="flex items-center justify-between font-mono">
          <span>Backend</span>
          <span className="text-emerald-400 font-semibold">FastAPI 0.115</span>
        </div>
        <div className="flex items-center justify-between font-mono mt-1">
          <span>Architecture</span>
          <span>Conflict-Aware</span>
        </div>
      </div>
    </aside>
  );
}
