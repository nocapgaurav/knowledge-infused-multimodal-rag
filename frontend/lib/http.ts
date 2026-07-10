import axios, { type AxiosError, type AxiosInstance } from "axios";

import { config } from "@/lib/config";
import { AppError, type AppErrorKind } from "@/types/errors";

/**
 * The single Axios instance every service is built on. Nothing outside
 * `services/` imports this -- see the `no-restricted-imports` ESLint rule
 * for `axios`, which enforces that at lint time.
 */
export function createHttpClient(timeoutMs: number = config.api.timeoutMs): AxiosInstance {
  return axios.create({
    baseURL: config.api.baseUrl,
    timeout: timeoutMs,
    headers: { Accept: "application/json" },
  });
}

export const http = createHttpClient();

/** Normalizes any thrown error from an HTTP call into an `AppError` with a
 * consistent, UI-safe classification -- never a raw Axios/network error
 * reaches a component or a rendered message. */
export function toAppError(error: unknown, fallbackMessage: string): AppError {
  if (error instanceof AppError) return error;

  if (axios.isAxiosError(error)) {
    return fromAxiosError(error, fallbackMessage);
  }

  return new AppError(fallbackMessage, "unknown", null, error);
}

function fromAxiosError(error: AxiosError, fallbackMessage: string): AppError {
  if (error.code === "ECONNABORTED") {
    return new AppError("The request took too long to respond.", "timeout", null, error);
  }
  if (!error.response) {
    return new AppError(
      "Could not reach the server. Check your connection and try again.",
      "network",
      null,
      error,
    );
  }

  const status = error.response.status;
  const kind: AppErrorKind =
    status === 404
      ? "not_found"
      : status === 422
        ? "validation"
        : status >= 500
          ? "server"
          : "unknown";

  const detail = extractDetail(error.response.data);
  return new AppError(detail ?? fallbackMessage, kind, status, error);
}

function extractDetail(data: unknown): string | null {
  if (typeof data === "object" && data !== null && "detail" in data) {
    const detail = (data as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return null;
}
