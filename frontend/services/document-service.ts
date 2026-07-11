import { config } from "@/lib/config";
import { createHttpClient, http, toAppError } from "@/lib/http";
import type {
  BuildGraphResponseDto,
  DocumentStatusResponseDto,
  DocumentUploadResponseDto,
  EmbedDocumentResponseDto,
  IndexDocumentResponseDto,
  ParseDocumentResponseDto,
  RepresentDocumentResponseDto,
} from "@/types/api";

/**
 * Wraps Module 3-8's real endpoints exactly as they exist. Modules 1-11
 * expose no single "prepare this document" call -- preparation is five
 * separate synchronous POSTs the frontend must sequence itself
 * (parse -> represent -> embed -> index -> graph). `prepareDocument`
 * below is that sequencing, presented to the rest of the app as one
 * operation; it never invents a backend capability, it only calls what's
 * really there, in order.
 */

/** Preparation stages load real ML models server-side and can far exceed
 * the default request timeout on a cold backend -- same idiom as
 * `generation-service.ts`'s dedicated client. */
const preparationHttp = createHttpClient(config.api.preparationTimeoutMs);

export async function uploadDocument(file: File): Promise<DocumentUploadResponseDto> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await preparationHttp.post<DocumentUploadResponseDto>("/documents", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  } catch (error) {
    throw toAppError(error, "Could not upload this document.");
  }
}

export async function getDocumentStatus(documentId: string): Promise<DocumentStatusResponseDto> {
  try {
    const response = await http.get<DocumentStatusResponseDto>(`/documents/${documentId}`);
    return response.data;
  } catch (error) {
    throw toAppError(error, "Could not check this document's status.");
  }
}

async function parseDocument(documentId: string): Promise<ParseDocumentResponseDto> {
  const response = await preparationHttp.post<ParseDocumentResponseDto>(
    `/documents/${documentId}/parse`,
  );
  return response.data;
}

async function representDocument(documentId: string): Promise<RepresentDocumentResponseDto> {
  const response = await preparationHttp.post<RepresentDocumentResponseDto>(
    `/documents/${documentId}/represent`,
  );
  return response.data;
}

async function embedDocument(documentId: string): Promise<EmbedDocumentResponseDto> {
  const response = await preparationHttp.post<EmbedDocumentResponseDto>(
    `/documents/${documentId}/embed`,
  );
  return response.data;
}

async function indexDocument(documentId: string): Promise<IndexDocumentResponseDto> {
  const response = await preparationHttp.post<IndexDocumentResponseDto>(
    `/documents/${documentId}/index`,
  );
  return response.data;
}

async function buildGraph(documentId: string): Promise<BuildGraphResponseDto> {
  const response = await preparationHttp.post<BuildGraphResponseDto>(
    `/documents/${documentId}/graph`,
  );
  return response.data;
}

export type PreparationStage = "parsing" | "representing" | "embedding" | "indexing" | "graphing";

const PREPARATION_SEQUENCE: {
  stage: PreparationStage;
  run: (documentId: string) => Promise<unknown>;
}[] = [
  { stage: "parsing", run: parseDocument },
  { stage: "representing", run: representDocument },
  { stage: "embedding", run: embedDocument },
  { stage: "indexing", run: indexDocument },
  { stage: "graphing", run: buildGraph },
];

/** Runs the full preparation pipeline in order, reporting each stage as it
 * starts. Throws an `AppError` (never a raw backend exception) on the
 * first failing stage -- the caller decides how to present that as
 * "Preparing failed" without ever naming the stage to the user. */
export async function prepareDocument(
  documentId: string,
  onStageStart?: (stage: PreparationStage) => void,
): Promise<void> {
  for (const { stage, run } of PREPARATION_SEQUENCE) {
    onStageStart?.(stage);
    try {
      await run(documentId);
    } catch (error) {
      throw toAppError(error, "Preparing this document failed.");
    }
  }
}
