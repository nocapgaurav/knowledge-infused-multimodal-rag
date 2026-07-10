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
      modality: group.primary.candidate.modality,
      text: group.primary.candidate.text,
      groupId: group.group_id,
      isPrimary: true,
    },
    ...group.supporting.map((member) => ({
      knowledgeUnitId: member.candidate.knowledge_unit_id,
      label: labelForCandidate(member.candidate.knowledge_unit_id, bundle),
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
