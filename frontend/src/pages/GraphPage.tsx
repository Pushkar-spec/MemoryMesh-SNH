import React, { useState, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
  type NodeMouseHandler,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Share2, RefreshCw, Filter, Layers } from 'lucide-react';
import { useGraph, useCatalog } from '../api/hooks';
import { EntityNode } from '../components/graph/EntityNode';
import { MemoryNode } from '../components/graph/MemoryNode';
import { GraphDetailPanel } from '../components/graph/GraphDetailPanel';
import { ProvenanceDrawer } from '../components/memories/ProvenanceDrawer';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorState } from '../components/common/ErrorState';
import { EmptyState } from '../components/common/EmptyState';

// Register node types outside component to prevent re-creation
const nodeTypes: NodeTypes = {
  entity: EntityNode,
  memory: MemoryNode,
};

import {
  transformGraphResponse,
} from '../lib/graphLayout';

export function GraphPage(): React.JSX.Element {
  const [selectedEntity, setSelectedEntity] = useState<string>('');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [inspectMemoryId, setInspectMemoryId] = useState<number | null>(null);

  const { data: catalog } = useCatalog();
  const {
    data: graphData,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useGraph(selectedEntity ? { entity: selectedEntity } : undefined);

  const apiNodes = useMemo(() => graphData?.nodes ?? [], [graphData?.nodes]);
  const apiEdges = useMemo(() => graphData?.edges ?? [], [graphData?.edges]);

  const { nodes: rfNodes, edges: rfEdges } = useMemo(
    () => transformGraphResponse({ nodes: apiNodes, edges: apiEdges }),
    [apiNodes, apiEdges]
  );

  const selectedNode = useMemo(
    () => apiNodes.find((n) => n.id === selectedNodeId) ?? null,
    [apiNodes, selectedNodeId]
  );

  const onNodeClick: NodeMouseHandler = (_, node) => {
    setSelectedNodeId(node.id);
  };

  const badgeClass =
    'px-2 py-0.5 rounded-full text-xs font-semibold bg-slate-800 ' +
    'text-slate-300 border border-slate-700';

  const refreshBtnClass =
    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 ' +
    'hover:bg-slate-700 text-slate-200 text-xs font-medium border ' +
    'border-slate-700 transition-colors cursor-pointer disabled:opacity-50';

  const legendClass =
    'flex flex-wrap items-center gap-3 text-xs bg-slate-900/60 p-2.5 ' +
    'rounded-lg border border-slate-800 shrink-0';

  return (
    <div className="space-y-4 h-[calc(100vh-8rem)] flex flex-col">
      {/* Header & Controls */}
      <div
        className={
          'flex flex-wrap items-center justify-between gap-4 border-b ' +
          'border-slate-800 pb-3 shrink-0'
        }
      >
        <div>
          <div className="flex items-center gap-2">
            <Share2 className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">Dependency Graph</h2>
            {apiNodes.length > 0 && (
              <span className={badgeClass}>
                {apiNodes.length} nodes · {apiEdges.length} edges
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Interactive React Flow visualization of memories, derivations, disputes, and relations.
          </p>
        </div>

        {/* Filters and Actions */}
        <div className="flex items-center gap-3">
          <div
            className={
              'flex items-center gap-2 bg-slate-900 border border-slate-800 ' +
              'rounded-lg px-2.5 py-1.5'
            }
          >
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

          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className={refreshBtnClass}
            title="Refresh graph"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Relationship Type Legend */}
      <div className={legendClass}>
        <span className="text-slate-400 font-semibold flex items-center gap-1">
          <Layers className="w-3.5 h-3.5" />
          Relations:
        </span>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-purple-400" />
          <span className="text-slate-300">supersedes</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
          <span className="text-slate-300">contradicts</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
          <span className="text-slate-300">depends_on</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
          <span className="text-slate-300">refines</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
          <span className="text-slate-400">about</span>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div
        className={
          'flex-1 relative rounded-xl border border-slate-800 bg-slate-950 ' +
          'overflow-hidden flex'
        }
      >
        {isLoading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/80">
            <LoadingSpinner label="Loading dependency graph..." />
          </div>
        )}

        {isError && (
          <div className="absolute inset-0 z-10 flex items-center justify-center p-6">
            <ErrorState
              error={error}
              title="Failed to load graph"
              onRetry={() => refetch()}
            />
          </div>
        )}

        {!isLoading && !isError && apiNodes.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center p-6">
            <EmptyState
              icon={Share2}
              title="No Graph Nodes Found"
              description="Ingest messages to populate entities and memories in the graph."
            />
          </div>
        )}

        {/* React Flow Container */}
        <div className="flex-1 h-full">
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            nodeTypes={nodeTypes}
            onNodeClick={onNodeClick}
            fitView
            minZoom={0.2}
            maxZoom={1.5}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#334155" gap={20} size={1} />
            <Controls
              className={
                'bg-slate-900 border border-slate-700 rounded text-slate-200 fill-slate-200'
              }
            />
            <MiniMap
              nodeColor={(node) => (node.type === 'entity' ? '#6366f1' : '#3b82f6')}
              className="bg-slate-900 border border-slate-800 rounded-lg"
              maskColor="rgba(15, 23, 42, 0.7)"
            />
          </ReactFlow>
        </div>

        {/* Node Detail Drawer / Panel */}
        {selectedNode && (
          <div className="absolute right-0 top-0 bottom-0 z-20">
            <GraphDetailPanel
              node={selectedNode}
              edges={apiEdges}
              onClose={() => setSelectedNodeId(null)}
              onInspectProvenance={(id) => setInspectMemoryId(id)}
            />
          </div>
        )}
      </div>

      {/* Provenance Drawer */}
      <ProvenanceDrawer
        memoryId={inspectMemoryId}
        onClose={() => setInspectMemoryId(null)}
      />
    </div>
  );
}
