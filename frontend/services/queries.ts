import { useQuery } from "@tanstack/react-query";

import { getBackendHealth } from "@/services/health-service";
import { retrieveEvidence } from "@/services/retrieval-service";

/** TanStack Query hooks -- the only place components read server state
 * from (Phase 2B: components never call a service's raw function
 * directly for anything that should be cached/retried/loading-tracked). */
export function useBackendHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getBackendHealth,
    refetchInterval: 30_000,
    retry: 0,
  });
}

/** Related Content (Phase 3/4B) is user-initiated, so this only runs
 * when `enabled` -- it calls Module 9's real `/retrieve` endpoint
 * directly (rather than reusing a `/generate` response) because only
 * the full `EvidenceBundle` carries `discovery_method`, which is how
 * "related via the graph" is distinguished from "matched directly." */
export function useRelatedContent(documentId: string, query: string | null) {
  return useQuery({
    queryKey: ["related-content", documentId, query],
    queryFn: () => retrieveEvidence(documentId, query as string),
    enabled: query !== null && query.length > 0,
    staleTime: 5 * 60_000,
  });
}
