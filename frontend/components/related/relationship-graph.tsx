"use client";

import { useMemo } from "react";
import ReactFlow, { Background, Controls, MiniMap, type Edge, type Node } from "reactflow";

import "reactflow/dist/style.css";

import { buildRelationshipGraph } from "@/utils/build-relationship-graph";
import type { RetrievalCandidateDto } from "@/types/api";

const DEPTH_COLUMN_WIDTH = 220;
const ROW_HEIGHT = 90;

/**
 * The Relationship Viewer (Phase 3 feature component; React Flow per the
 * Phase 2A stack). Renders only real graph-traversal data already on the
 * candidates -- see `utils/build-relationship-graph.ts` -- never a
 * speculative layout invented for visual effect.
 */
export function RelationshipGraph({ candidates }: { candidates: RetrievalCandidateDto[] }) {
  const { nodes, edges } = useMemo(() => {
    const graph = buildRelationshipGraph(candidates);
    const rowByDepth = new Map<number, number>();

    const flowNodes: Node[] = graph.nodes.map((node) => {
      const row = rowByDepth.get(node.depth) ?? 0;
      rowByDepth.set(node.depth, row + 1);
      return {
        id: node.id,
        position: { x: node.depth * DEPTH_COLUMN_WIDTH, y: row * ROW_HEIGHT },
        data: { label: node.label },
        style: node.isKnownCandidate
          ? { borderColor: "var(--evidence)", color: "var(--foreground)" }
          : { opacity: 0.6 },
      };
    });

    const flowEdges: Edge[] = graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      animated: false,
    }));

    return { nodes: flowNodes, edges: flowEdges };
  }, [candidates]);

  if (nodes.length === 0) {
    return null;
  }

  return (
    <div className="h-full w-full">
      <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
        <Background />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable className="!bg-card" />
      </ReactFlow>
    </div>
  );
}
