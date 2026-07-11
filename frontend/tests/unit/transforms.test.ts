import { describe, expect, it } from "vitest";

import {
  answerViewModelFromGroundedResponse,
  evidenceItemsFromBundle,
} from "@/services/transforms";
import type { EvidenceBundleDto, GroundedResponseDto } from "@/types/api";

function buildBundle(): EvidenceBundleDto {
  return {
    document_id: "doc-1",
    query: "What are the main results?",
    candidates: [
      {
        knowledge_unit_id: "ku-1",
        document_id: "doc-1",
        section_id: null,
        modality: "text",
        text: "primary text",
        retrieval_context: "Section: 1. Introduction",
        page_numbers: [2],
        asset_uri: null,
        reading_order: 0,
        citation_count: 0,
        dense_similarity: 0.9,
        discovery_method: "dense_retrieval",
        graph_path: { hops: [] },
      },
      {
        knowledge_unit_id: "ku-2",
        document_id: "doc-1",
        section_id: null,
        modality: "table",
        text: "supporting text",
        retrieval_context: "Table 1",
        page_numbers: [3],
        asset_uri: null,
        reading_order: 1,
        citation_count: 0,
        dense_similarity: 0.8,
        discovery_method: "dense_retrieval",
        graph_path: { hops: [] },
      },
    ],
    evidence_groups: [
      {
        group_id: "group-1",
        primary: {
          candidate: {
            knowledge_unit_id: "ku-1",
            document_id: "doc-1",
            section_id: null,
            modality: "text",
            text: "primary text",
            retrieval_context: "Section: 1. Introduction",
            page_numbers: [2],
            asset_uri: null,
            reading_order: 0,
            citation_count: 0,
            dense_similarity: 0.9,
            discovery_method: "dense_retrieval",
            graph_path: { hops: [] },
          },
          ranking: {
            signals: [{ name: "dense_similarity", raw_value: 0.9, rank: 1 }],
            fused_score: 0.9,
            final_rank: 1,
          },
        },
        supporting: [
          {
            candidate: {
              knowledge_unit_id: "ku-2",
              document_id: "doc-1",
              section_id: null,
              modality: "table",
              text: "supporting text",
              retrieval_context: "Table 1",
              page_numbers: [3],
              asset_uri: null,
              reading_order: 1,
              citation_count: 0,
              dense_similarity: 0.8,
              discovery_method: "dense_retrieval",
              graph_path: { hops: [] },
            },
            ranking: {
              signals: [{ name: "dense_similarity", raw_value: 0.8, rank: 2 }],
              fused_score: 0.8,
              final_rank: 2,
            },
          },
        ],
        modalities: ["text", "table"],
      },
    ],
    trace: { phases: [], dropped: [] },
    manifest: {
      document_id: "doc-1",
      query: "What are the main results?",
      retrieval_version: "1.0",
      retrieval_strategy_version: "1.0",
      representation_version: "repr-1",
      embedding_version: "embed-1",
      graph_version: "1.0",
      statistics: {
        candidates_generated: 2,
        candidates_expanded: 0,
        candidates_scored: 2,
        evidence_groups: 1,
        evidence_items: 2,
        duration_ms: 5,
      },
      created_at: "2026-01-01T00:00:00Z",
    },
  };
}

describe("evidenceItemsFromBundle", () => {
  it("flattens every evidence group into primary-then-supporting order", () => {
    const items = evidenceItemsFromBundle(buildBundle());

    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({
      knowledgeUnitId: "ku-1",
      isPrimary: true,
      groupId: "group-1",
    });
    expect(items[1]).toMatchObject({
      knowledgeUnitId: "ku-2",
      isPrimary: false,
      groupId: "group-1",
    });
  });

  it("never exposes a raw knowledge_unit_id as the display label", () => {
    const items = evidenceItemsFromBundle(buildBundle());

    for (const item of items) {
      expect(item.label).not.toBe(item.knowledgeUnitId);
    }
  });
});

function buildGroundedResponse(): GroundedResponseDto {
  return {
    document_id: "doc-1",
    query: "What are the main results?",
    answer: "The results show improvement [KU1].",
    executive_summary: "Improvement observed.",
    supporting_evidence: [
      {
        label: "KU1",
        knowledge_unit_id: "ku-1",
        text: "primary text",
        modality: "text",
        display_label: "Section: 1. Introduction",
        page_numbers: [2],
        relevance: 0.9,
        discovery: "Matched your question directly",
      },
    ],
    resolved_citations: [
      {
        label: "KU1",
        knowledge_unit_id: "ku-1",
        text_excerpt: "primary text",
        display_label: "Section: 1. Introduction",
        page_numbers: [2],
      },
    ],
    limitations: [],
    references: ["[KU1] primary text"],
    warnings: [],
    confidence: 0.9,
    answer_status: "sufficient_evidence",
    generation_metadata: {},
    prompt_version: "1.0",
    model_name: "qwen2.5:7b-instruct",
    model_version: "digest-1",
    generation_trace: { phases: [] },
    generation_statistics: {
      context_sections_used: 1,
      context_sections_dropped: 0,
      claims_total: 1,
      claims_grounded: 1,
      citations_resolved: 1,
      citations_unresolved: 0,
      prompt_tokens: 10,
      completion_tokens: 5,
      duration_ms: 5,
    },
    answer_provenance: {
      document_id: "doc-1",
      retrieval_version: "1.0",
      retrieval_strategy_version: "1.0",
      representation_version: "repr-1",
      embedding_version: "embed-1",
      graph_version: "1.0",
      knowledge_unit_ids: ["ku-1"],
      evidence_bundle_checksum: "checksum-1",
    },
  };
}

describe("answerViewModelFromGroundedResponse", () => {
  it("carries the citation label through, not the raw knowledge unit id", () => {
    const answer = answerViewModelFromGroundedResponse(buildGroundedResponse());

    expect(answer.citations).toEqual([
      {
        label: "KU1",
        knowledgeUnitId: "ku-1",
        textExcerpt: "primary text",
        displayLabel: "Section: 1. Introduction",
        pageNumbers: [2],
      },
    ]);
    expect(answer.confidence).toBe(0.9);
    expect(answer.answerStatus).toBe("sufficient_evidence");
  });
});
