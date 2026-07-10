import { http, toAppError } from "@/lib/http";
import type { HealthResponseDto } from "@/types/api";

/** Wraps the backend's `/health` endpoint, used for the Settings screen's
 * "Backend Connection Status" (Phase 4C). */
export async function getBackendHealth(): Promise<HealthResponseDto> {
  try {
    const response = await http.get<HealthResponseDto>("/health");
    return response.data;
  } catch (error) {
    throw toAppError(error, "Could not reach the backend.");
  }
}
