import type { RetrievalCandidateDto } from "@/types/api";

/**
 * Builds a node/edge graph purely from real graph-traversal data already
 * present on graph-expanded candidates (`graph_path.hops`) -- never a
 * speculative or fabricated relationship (Phase 3: "Only relationships
 * discovered from the uploaded document should appear"). Kept as a pure
 * function, independent of React Flow's types, so it stays unit-testable
 * without mounting a canvas.
 */

export interface GraphNodeData {
  id: string;
  label: string;
  isKnownCandidate: boolean;
  depth: number;
}

export interface GraphEdgeData {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface RelationshipGraphData {
  nodes: GraphNodeData[];
  edges: GraphEdgeData[];
}

export function buildRelationshipGraph(candidates: RetrievalCandidateDto[]): RelationshipGraphData {
  const candidatesById = new Map(
    candidates.map((candidate) => [candidate.knowledge_unit_id, candidate]),
  );
  const nodes = new Map<string, GraphNodeData>();
  const edges: GraphEdgeData[] = [];

  const graphExpandedCandidates = candidates.filter(
    (candidate) =>
      candidate.discovery_method === "graph_expansion" && candidate.graph_path.hops.length > 0,
  );

  for (const candidate of graphExpandedCandidates) {
    candidate.graph_path.hops.forEach((hop, depth) => {
      addNode(nodes, hop.source_id, candidatesById, depth);
      addNode(nodes, hop.target_id, candidatesById, depth + 1);

      const [source, target] =
        hop.direction === "incoming"
          ? [hop.target_id, hop.source_id]
          : [hop.source_id, hop.target_id];
      edges.push({
        id: `${candidate.knowledge_unit_id}-${depth}`,
        source,
        target,
        label: hop.relationship_type,
      });
    });
  }

  return { nodes: Array.from(nodes.values()), edges };
}

function addNode(
  nodes: Map<string, GraphNodeData>,
  id: string,
  candidatesById: Map<string, RetrievalCandidateDto>,
  depth: number,
): void {
  if (nodes.has(id)) return;
  const candidate = candidatesById.get(id);
  nodes.set(id, {
    id,
    label: candidate ? labelFor(candidate) : "Connected content",
    isKnownCandidate: candidate !== undefined,
    depth,
  });
}

function labelFor(candidate: RetrievalCandidateDto): string {
  if (candidate.modality === "figure") return "Figure";
  if (candidate.modality === "table") return "Table";
  return candidate.text.slice(0, 40) + (candidate.text.length > 40 ? "..." : "");
}
