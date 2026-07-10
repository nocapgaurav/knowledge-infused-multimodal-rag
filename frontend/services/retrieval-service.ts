import { http, toAppError } from "@/lib/http";
import { evidenceItemsFromBundle } from "@/services/transforms";
import type { EvidenceBundleDto, RetrieveEvidenceRequestDto } from "@/types/api";
import type { EvidenceItem } from "@/types/view-models";

export interface RetrieveEvidenceResult {
  bundle: EvidenceBundleDto;
  evidenceItems: EvidenceItem[];
}

/** Wraps Module 9's real `/documents/{id}/retrieve` endpoint. */
export async function retrieveEvidence(
  documentId: string,
  query: string,
): Promise<RetrieveEvidenceResult> {
  try {
    const body: RetrieveEvidenceRequestDto = { query };
    const response = await http.post<EvidenceBundleDto>(`/documents/${documentId}/retrieve`, body);
    return { bundle: response.data, evidenceItems: evidenceItemsFromBundle(response.data) };
  } catch (error) {
    throw toAppError(error, "Could not retrieve evidence for this question.");
  }
}
