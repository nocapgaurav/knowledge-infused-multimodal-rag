import { config } from "@/lib/config";
import { createHttpClient, toAppError } from "@/lib/http";
import { answerViewModelFromGroundedResponse } from "@/services/transforms";
import type { GenerateAnswerRequestDto, GroundedResponseDto } from "@/types/api";
import type { AnswerViewModel } from "@/types/view-models";

const generationHttp = createHttpClient(config.api.generationTimeoutMs);

export interface GenerateAnswerResult {
  response: GroundedResponseDto;
  answer: AnswerViewModel;
}

/** Wraps Module 10's real `/documents/{id}/generate` endpoint. Uses a
 * longer timeout than the default client -- a real local LLM call, not a
 * quick lookup (see Module 10/11's own verification: 5-20s typical). */
export async function generateAnswer(
  documentId: string,
  query: string,
): Promise<GenerateAnswerResult> {
  try {
    const body: GenerateAnswerRequestDto = { query };
    const response = await generationHttp.post<GroundedResponseDto>(
      `/documents/${documentId}/generate`,
      body,
    );
    return { response: response.data, answer: answerViewModelFromGroundedResponse(response.data) };
  } catch (error) {
    throw toAppError(error, "Could not generate an answer for this question.");
  }
}
