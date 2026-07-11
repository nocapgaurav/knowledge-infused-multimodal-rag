/**
 * DTO -> view-model transforms. Kept in one place so every service builds
 * presentation models the same way -- never duplicated per call site.
 */

import type { EvidenceBundleDto, GroundedResponseDto } from "@/types/api";
import type { AnswerViewModel, Citation, EvidenceItem } from "@/types/view-models";

export function evidenceItemsFromBundle(bundle: EvidenceBundleDto): EvidenceItem[] {
  return bundle.evidence_groups.flatMap((group) => [
    {
      knowledgeUnitId: group.primary.candidate.knowledge_unit_id,
      label: labelForCandidate(group.primary.candidate.knowledge_unit_id, bundle),
      displayLabel: group.primary.candidate.retrieval_context ?? undefined,
      pageNumbers: group.primary.candidate.page_numbers?.length
        ? group.primary.candidate.page_numbers
        : undefined,
      boundingBoxes: group.primary.candidate.bounding_boxes?.length
        ? group.primary.candidate.bounding_boxes
        : undefined,
      modality: group.primary.candidate.modality,
      text: group.primary.candidate.text,
      groupId: group.group_id,
      isPrimary: true,
    },
    ...group.supporting.map((member) => ({
      knowledgeUnitId: member.candidate.knowledge_unit_id,
      label: labelForCandidate(member.candidate.knowledge_unit_id, bundle),
      displayLabel: member.candidate.retrieval_context ?? undefined,
      pageNumbers: member.candidate.page_numbers?.length
        ? member.candidate.page_numbers
        : undefined,
      boundingBoxes: member.candidate.bounding_boxes?.length
        ? member.candidate.bounding_boxes
        : undefined,
      modality: member.candidate.modality,
      text: member.candidate.text,
      groupId: group.group_id,
      isPrimary: false,
    })),
  ]);
}

/** Retrieval never assigns citation labels (that's Module 10's job) --
 * fall back to a stable, reading-order-based label so evidence is at
 * least referenceable before an answer exists. */
function labelForCandidate(knowledgeUnitId: string, bundle: EvidenceBundleDto): string {
  const index = bundle.candidates.findIndex((c) => c.knowledge_unit_id === knowledgeUnitId);
  return `E${index >= 0 ? index + 1 : "?"}`;
}

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
