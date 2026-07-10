/**
 * Presentation-shaped types every component actually consumes. Derived
 * from `types/api.ts` DTOs but never identical to them -- these represent
 * what a screen needs to render, not what the backend happened to persist
 * (Phase 2B: "Frontend types should represent presentation requirements
 * rather than storage structures").
 */

import type { AnswerStatus, ChunkModality } from "@/types/api";

// ---------------------------------------------------------------------------
// Documents (client-tracked -- see module12-backend-integration-gaps memory:
// Modules 1-11 expose no list/delete endpoint, so this list lives entirely
// in the browser).
// ---------------------------------------------------------------------------

/** The only three states ever shown to a user (Phase 4A: "Never expose
 * backend stage names"). Internal pipeline bookkeeping is a separate,
 * never-rendered-directly concern -- see `store/document-store.ts`. */
export type DocumentDisplayStatus = "preparing" | "ready" | "failed";

export interface DocumentSummary {
  documentId: string;
  filename: string;
  uploadedAt: string;
  status: DocumentDisplayStatus;
  /** Present only when `status === "failed"`; user-facing, never a raw
   * backend exception string. */
  failureReason?: string;
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

export interface EvidenceItem {
  knowledgeUnitId: string;
  /** The citation label as the model used it (e.g. "KU2") -- the only
   * user-facing identifier; the raw knowledge_unit_id is never displayed. */
  label: string;
  modality: ChunkModality;
  text: string;
  groupId: string;
  isPrimary: boolean;
  /** True once real page-location has been resolved client-side by
   * searching for `text` inside the locally-held PDF (see pdf sync
   * utilities) -- absent while unresolved or when no PDF is available. */
  pdfPageNumber?: number;
}

export interface Citation {
  label: string;
  knowledgeUnitId: string;
  textExcerpt: string;
}

// ---------------------------------------------------------------------------
// Conversation
// ---------------------------------------------------------------------------

export interface ConversationTurn {
  id: string;
  question: string;
  /** Stamped by the client on receipt -- `GroundedResponse` carries no
   * timestamp of its own. */
  askedAt: string;
  status: "pending" | "complete" | "failed";
  answer?: AnswerViewModel;
  failureReason?: string;
}

export interface AnswerViewModel {
  answer: string;
  executiveSummary: string;
  confidence: number;
  answerStatus: AnswerStatus;
  citations: Citation[];
  evidenceItems: EvidenceItem[];
  limitations: string[];
  warnings: string[];
  references: string[];
}
