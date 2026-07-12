import { useQuery } from "@tanstack/react-query";

import { getBackendHealth } from "@/services/health-service";

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
