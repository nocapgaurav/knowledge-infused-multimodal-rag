/**
 * Centralized application configuration (Phase 2B: "Configuration should
 * never be scattered across the codebase"). Every environment-dependent
 * value is read here, once, and nowhere else -- business logic never
 * inspects `process.env` or `NODE_ENV` directly.
 */

type Environment = "development" | "test" | "production";

function resolveEnvironment(): Environment {
  if (process.env.NODE_ENV === "test") return "test";
  if (process.env.NODE_ENV === "production") return "production";
  return "development";
}

export const config = {
  app: {
    name: "Research Workspace",
    version: process.env.NEXT_PUBLIC_APP_VERSION ?? "0.1.0",
  },
  environment: resolveEnvironment(),
  api: {
    /** Base URL of the Modules 1-11 FastAPI backend. Never hardcoded at
     * call sites -- every service reads it from here. */
    baseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
    timeoutMs: 30_000,
    /** Generation calls run a real local LLM and can legitimately take
     * much longer than a typical request; measured at 5-20s during
     * Module 10/11's own verification. */
    generationTimeoutMs: 120_000,
    /** Preparation stages (parse/represent/embed/index/graph) load real
     * ML models server-side; on a cold backend the first embed alone can
     * exceed two minutes while weights load. Observed live: an upload
     * during backend cold-start failed at the embedding stage with a
     * client-side timeout at the default 30s. */
    preparationTimeoutMs: 300_000,
  },
  upload: {
    maxSizeBytes: 50 * 1024 * 1024,
    acceptedMimeTypes: ["application/pdf"] as string[],
  },
  featureFlags: {
    /** Relationship visualization (Phase 3/4B: optional, user-initiated,
     * never the default view). Kept behind a flag so it can be disabled
     * entirely without touching component code. */
    relatedContentGraphView: true,
  },
} as const;

export type AppConfig = typeof config;
