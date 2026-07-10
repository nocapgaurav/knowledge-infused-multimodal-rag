"use client";

import { useEffect } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { TYPOGRAPHY } from "@/constants/typography";

/**
 * Next's route-level error boundary. Distinct from
 * `components/feedback/error-boundary.tsx` (which isolates individual
 * workspace panels) -- this one catches anything that escapes all of
 * them, for a whole route segment (Phase 4C: explain, recover, never a
 * stack trace).
 */
export default function RouteError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") {
      console.error(error);
    }
  }, [error]);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
      <AlertTriangle className="text-error size-8" aria-hidden="true" />
      <p className={TYPOGRAPHY.workspaceTitle}>Something went wrong</p>
      <p className={`${TYPOGRAPHY.body} text-muted-foreground max-w-sm`}>
        This page ran into a problem. You can try again without losing your other documents.
      </p>
      <Button onClick={reset}>
        <RotateCcw className="size-4" />
        Try again
      </Button>
    </div>
  );
}
