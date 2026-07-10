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

export type DiscoveryMethod = "dense_retrieval" | "graph_expansion";

export type TraversalDirection = "outgoing" | "incoming";

export type AnswerStatus =
  "sufficient_evidence" | "partially_sufficient_evidence" | "insufficient_evidence";

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
// Retrieval (Module 9)
// ---------------------------------------------------------------------------

export interface TraversalHopDto {
  source_id: string;
  target_id: string;
  relationship_type: string;
  direction: TraversalDirection;
}

export interface GraphPathDto {
  hops: TraversalHopDto[];
}

export interface RetrievalCandidateDto {
  knowledge_unit_id: string;
  document_id: string;
  section_id: string | null;
  modality: ChunkModality;
  text: string;
  /** Backend-internal storage path, not a fetchable URL -- see
   * module12-backend-integration-gaps memory. Never rendered as an <img> src. */
  asset_uri: string | null;
  reading_order: number;
  citation_count: number;
  dense_similarity: number | null;
  discovery_method: DiscoveryMethod;
  graph_path: GraphPathDto;
}

export interface SignalScoreDto {
  name: string;
  raw_value: number;
  rank: number;
}

export interface RankingExplanationDto {
  signals: SignalScoreDto[];
  fused_score: number;
  final_rank: number;
}

export interface ScoredCandidateDto {
  candidate: RetrievalCandidateDto;
  ranking: RankingExplanationDto;
}

export interface EvidenceGroupDto {
  group_id: string;
  primary: ScoredCandidateDto;
  supporting: ScoredCandidateDto[];
  modalities: ChunkModality[];
}

export interface RetrievalPhaseTraceDto {
  phase: string;
  input_count: number;
  output_count: number;
  duration_ms: number;
  notes: string[];
}

export interface DroppedCandidateDto {
  knowledge_unit_id: string;
  phase: string;
  reason: string;
}

export interface RetrievalTraceDto {
  phases: RetrievalPhaseTraceDto[];
  dropped: DroppedCandidateDto[];
}

export interface RetrievalStatisticsDto {
  candidates_generated: number;
  candidates_expanded: number;
  candidates_scored: number;
  evidence_groups: number;
  evidence_items: number;
  duration_ms: number;
}

export interface RetrievalManifestDto {
  document_id: string;
  query: string;
  retrieval_version: string;
  retrieval_strategy_version: string;
  representation_version: string;
  embedding_version: string;
  graph_version: string;
  statistics: RetrievalStatisticsDto;
  created_at: string;
}

export interface EvidenceBundleDto {
  document_id: string;
  query: string;
  candidates: RetrievalCandidateDto[];
  evidence_groups: EvidenceGroupDto[];
  trace: RetrievalTraceDto;
  manifest: RetrievalManifestDto;
}

export interface RetrieveEvidenceRequestDto {
  query: string;
}

// ---------------------------------------------------------------------------
// Generation (Module 10)
// ---------------------------------------------------------------------------

export interface SupportingEvidenceItemDto {
  label: string;
  knowledge_unit_id: string;
  text: string;
  modality: ChunkModality;
}

export interface ResolvedCitationDto {
  label: string;
  knowledge_unit_id: string;
  text_excerpt: string;
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
