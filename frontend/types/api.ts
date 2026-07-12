/**
 * API DTOs: a 1:1 mirror of Modules 1-11's actual FastAPI response/request
 * schemas (verified directly against `create_app().openapi()`, not guessed).
 * Nothing here is a UI/view model -- see `types/view-models.ts` for the
 * presentation-shaped types components actually consume (Phase 2B: "Do not
 * mirror backend domain models unnecessarily... maintain clear separation
 * between API Types [and] UI Models").
 */

// ---------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------

export type UploadStatus = "UPLOADED" | "VALIDATING" | "READY_FOR_PARSING" | "FAILED";

export type ChunkModality = "text" | "table" | "figure";

export interface BoundingBoxDto {
  page_number: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type AnswerStatus =
  | "sufficient_evidence"
  | "partially_sufficient_evidence"
  /** Evidence was retrieved but no claim in this answer could be verified
   * against it -- a property of the answer, not the paper. */
  | "unverified_answer"
  /** Retrieval found nothing to answer from. */
  | "insufficient_evidence";

// ---------------------------------------------------------------------------
// Document lifecycle (Module 3: ingestion; Modules 4-8: pipeline stages)
// ---------------------------------------------------------------------------

export interface DocumentUploadResponseDto {
  document_id: string;
  upload_job_id: string;
  status: UploadStatus;
}

export interface DocumentStatusResponseDto {
  document_id: string;
  upload_job_id: string;
  status: UploadStatus;
  error_message: string | null;
  updated_at: string;
}

export interface ParseDocumentResponseDto {
  document_id: string;
  status: "PARSED";
}

export interface RepresentDocumentResponseDto {
  document_id: string;
  knowledge_units: number;
  relationships: number;
  status: "REPRESENTED";
}

export interface EmbedDocumentResponseDto {
  document_id: string;
  embeddings_generated: number;
  model: string;
  status: "EMBEDDED";
}

export interface IndexDocumentResponseDto {
  document_id: string;
  collection: string;
  indexed_vectors: number;
  status: "INDEXED";
}

export interface BuildGraphResponseDto {
  document_id: string;
  nodes: number;
  relationships: number;
  status: "GRAPH_CREATED";
}

// ---------------------------------------------------------------------------
// Generation (Module 10)
// ---------------------------------------------------------------------------

export interface SupportingEvidenceItemDto {
  label: string;
  knowledge_unit_id: string;
  text: string;
  modality: ChunkModality;
  /** Human-readable identity ("Figure 2", "Section: III. Methodology"). */
  display_label: string | null;
  page_numbers: number[];
  bounding_boxes: BoundingBoxDto[];
  /** Dense similarity when found by direct semantic match; null when
   * discovered through the knowledge graph. */
  relevance: number | null;
  /** Honest provenance note derived from real retrieval facts. */
  discovery: string | null;
}

export interface ResolvedCitationDto {
  label: string;
  knowledge_unit_id: string;
  text_excerpt: string;
  display_label: string | null;
  page_numbers: number[];
  bounding_boxes: BoundingBoxDto[];
  modality: ChunkModality;
}

export interface GenerationStatisticsDto {
  context_sections_used: number;
  context_sections_dropped: number;
  claims_total: number;
  claims_grounded: number;
  citations_resolved: number;
  citations_unresolved: number;
  prompt_tokens: number;
  completion_tokens: number;
  duration_ms: number;
}

export interface AnswerProvenanceDto {
  document_id: string;
  retrieval_version: string;
  retrieval_strategy_version: string;
  representation_version: string;
  embedding_version: string;
  graph_version: string;
  knowledge_unit_ids: string[];
  evidence_bundle_checksum: string;
}

export interface GenerationPhaseTraceDto {
  phase: string;
  input_count: number;
  output_count: number;
  duration_ms: number;
  notes: string[];
}

export interface GenerationTraceDto {
  phases: GenerationPhaseTraceDto[];
}

export interface GenerateAnswerRequestDto {
  query: string;
}

export interface GroundedResponseDto {
  document_id: string;
  query: string;
  answer: string;
  executive_summary: string;
  supporting_evidence: SupportingEvidenceItemDto[];
  resolved_citations: ResolvedCitationDto[];
  limitations: string[];
  references: string[];
  warnings: string[];
  confidence: number;
  answer_status: AnswerStatus;
  generation_metadata: Record<string, string>;
  prompt_version: string;
  model_name: string;
  model_version: string;
  generation_trace: GenerationTraceDto;
  generation_statistics: GenerationStatisticsDto;
  answer_provenance: AnswerProvenanceDto;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export interface HealthResponseDto {
  status: "ok";
}
