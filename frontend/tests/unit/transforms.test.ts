import { describe, expect, it } from "vitest";

import { answerViewModelFromGroundedResponse } from "@/services/transforms";
import type { GroundedResponseDto } from "@/types/api";

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
        bounding_boxes: [{ page_number: 2, x0: 10, y0: 20, x1: 100, y1: 40 }],
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
        bounding_boxes: [],
        modality: "text",
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
        modality: "text",
      },
    ]);
    expect(answer.confidence).toBe(0.9);
    expect(answer.answerStatus).toBe("sufficient_evidence");
  });
});
