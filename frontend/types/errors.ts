/**
 * Normalized application error shape. Every service catches raw Axios/
 * network errors and re-throws one of these instead, so components never
 * branch on Axios internals (Phase 2B: "Errors should be classified
 * consistently"; Phase 4C: named error categories).
 */
export type AppErrorKind =
  "network" | "timeout" | "validation" | "not_found" | "server" | "unknown";

export class AppError extends Error {
  readonly kind: AppErrorKind;
  readonly status: number | null;

  constructor(message: string, kind: AppErrorKind, status: number | null = null, cause?: unknown) {
    super(message, { cause });
    this.name = "AppError";
    this.kind = kind;
    this.status = status;
  }
}
