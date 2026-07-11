/**
 * Presentation-shaped types every component actually consumes. Derived
 * from `types/api.ts` DTOs but never identical to them -- these represent
 * what a screen needs to render, not what the backend happened to persist
 * (Phase 2B: "Frontend types should represent presentation requirements
 * rather than storage structures").
 */

import type { AnswerStatus, BoundingBoxDto, ChunkModality } from "@/types/api";

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
  /** The citation label as the model used it (e.g. "KU2") -- kept for
   * traceability to the answer's inline citations, but no longer the
   * primary identity a reader sees (that is `displayLabel`). */
  label: string;
  /** Human-readable identity ("Figure 2", "Abstract", "Section: III.
   * Methodology"), when the backend knows it. */
  displayLabel?: string;
  /** Source PDF page(s), from the parser's own bounding boxes. */
  pageNumbers?: number[];
  /** Exact source locations (top-left origin, PDF points at 1x zoom),
   * for in-PDF evidence highlighting. */
  boundingBoxes?: BoundingBoxDto[];
  /** Dense similarity to the question, when found by direct match. */
  relevance?: number;
  /** Honest provenance: how retrieval found this evidence. */
  discovery?: string;
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
  /** Human-readable identity of the cited evidence, when known. */
  displayLabel?: string;
  pageNumbers?: number[];
  boundingBoxes?: BoundingBoxDto[];
  modality?: ChunkModality;
}

/** The selection payload carried alongside the canonical
 * `openedEvidenceId`: everything the PDF viewer needs to locate this
 * evidence, snapshotted from whichever view was clicked (answer
 * citation, evidence card, or related item). The id remains the single
 * synchronization key across all views. */
export interface EvidenceTarget {
  text: string;
  displayLabel?: string;
  pageNumbers?: number[];
  boundingBoxes?: BoundingBoxDto[];
  /** Evidence type -- figures and tables highlight as regions (the
   * visual itself is the evidence), text as exact passages. */
  modality?: ChunkModality;
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
