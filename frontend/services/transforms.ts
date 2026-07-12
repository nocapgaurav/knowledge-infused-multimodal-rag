/**
 * DTO -> view-model transforms. Kept in one place so every service builds
 * presentation models the same way -- never duplicated per call site.
 */

import type { GroundedResponseDto } from "@/types/api";
import type { AnswerViewModel, Citation, EvidenceItem } from "@/types/view-models";

export function evidenceItemsFromGroundedResponse(response: GroundedResponseDto): EvidenceItem[] {
  return response.supporting_evidence.map((item) => ({
    knowledgeUnitId: item.knowledge_unit_id,
    label: item.label,
    displayLabel: item.display_label ?? undefined,
    pageNumbers: item.page_numbers?.length ? item.page_numbers : undefined,
    boundingBoxes: item.bounding_boxes?.length ? item.bounding_boxes : undefined,
    relevance: item.relevance ?? undefined,
    discovery: item.discovery ?? undefined,
    modality: item.modality,
    text: item.text,
    groupId: item.knowledge_unit_id,
    isPrimary: true,
  }));
}

export function citationsFromGroundedResponse(response: GroundedResponseDto): Citation[] {
  return response.resolved_citations.map((citation) => ({
    label: citation.label,
    knowledgeUnitId: citation.knowledge_unit_id,
    textExcerpt: citation.text_excerpt,
    displayLabel: citation.display_label ?? undefined,
    pageNumbers: citation.page_numbers?.length ? citation.page_numbers : undefined,
    boundingBoxes: citation.bounding_boxes?.length ? citation.bounding_boxes : undefined,
    modality: citation.modality,
  }));
}

export function answerViewModelFromGroundedResponse(
  response: GroundedResponseDto,
): AnswerViewModel {
  return {
    answer: response.answer,
    executiveSummary: response.executive_summary,
    confidence: response.confidence,
    answerStatus: response.answer_status,
    citations: citationsFromGroundedResponse(response),
    evidenceItems: evidenceItemsFromGroundedResponse(response),
    limitations: response.limitations,
    warnings: response.warnings,
    references: response.references,
  };
}
