import { describe, expect, it } from "vitest";

import { buildRelationshipGraph } from "@/utils/build-relationship-graph";
import type { RetrievalCandidateDto } from "@/types/api";

function candidate(overrides: Partial<RetrievalCandidateDto>): RetrievalCandidateDto {
  return {
    knowledge_unit_id: "ku-1",
    document_id: "doc-1",
    section_id: null,
    modality: "text",
    retrieval_context: null,
    bounding_boxes: [],
    page_numbers: [],
    text: "some evidence text",
    asset_uri: null,
    reading_order: 0,
    citation_count: 0,
    dense_similarity: 0.9,
    discovery_method: "dense_retrieval",
    graph_path: { hops: [] },
    ...overrides,
  };
}

describe("buildRelationshipGraph", () => {
  it("ignores candidates discovered by dense retrieval alone", () => {
    const graph = buildRelationshipGraph([candidate({})]);

    expect(graph.nodes).toHaveLength(0);
    expect(graph.edges).toHaveLength(0);
  });

  it("builds nodes and edges only from real graph_path hops", () => {
    const expanded = candidate({
      knowledge_unit_id: "ku-2",
      discovery_method: "graph_expansion",
      graph_path: {
        hops: [
          {
            source_id: "ku-1",
            target_id: "ku-2",
            relationship_type: "CITES",
            direction: "outgoing",
          },
        ],
      },
    });

    const graph = buildRelationshipGraph([candidate({}), expanded]);

    expect(graph.nodes.map((n) => n.id).sort()).toEqual(["ku-1", "ku-2"]);
    expect(graph.edges).toEqual([{ id: "ku-2-0", source: "ku-1", target: "ku-2", label: "CITES" }]);
  });

  it("reverses source/target for incoming hops", () => {
    const expanded = candidate({
      knowledge_unit_id: "ku-2",
      discovery_method: "graph_expansion",
      graph_path: {
        hops: [
          {
            source_id: "ku-2",
            target_id: "ku-1",
            relationship_type: "CITES",
            direction: "incoming",
          },
        ],
      },
    });

    const graph = buildRelationshipGraph([expanded]);

    expect(graph.edges[0]).toMatchObject({ source: "ku-1", target: "ku-2" });
  });

  it("labels nodes with no matching candidate as generic connected content, never a raw id", () => {
    const expanded = candidate({
      knowledge_unit_id: "ku-2",
      discovery_method: "graph_expansion",
      graph_path: {
        hops: [
          {
            source_id: "unknown-node-id",
            target_id: "ku-2",
            relationship_type: "NEXT",
            direction: "outgoing",
          },
        ],
      },
    });

    const graph = buildRelationshipGraph([expanded]);
    const unknownNode = graph.nodes.find((n) => n.id === "unknown-node-id");

    expect(unknownNode?.label).toBe("Connected content");
    expect(unknownNode?.isKnownCandidate).toBe(false);
  });
});
