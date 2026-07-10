import Link from "next/link";
import { Compass } from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { TYPOGRAPHY } from "@/constants/typography";

/**
 * Next.js only renders a route group's own `not-found.tsx` for paths
 * that at least resolve into that group; a completely unmatched path
 * (outside every group) falls back to this root-level file instead. It
 * renders the same shell so users never land on an unstyled 404 (Phase
 * 4A: "Users should never become disoriented").
 */
export default function RootNotFound() {
  return (
    <AppShell>
      <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <Compass className="text-muted-foreground size-8" aria-hidden="true" />
        <p className={TYPOGRAPHY.workspaceTitle}>This page doesn&apos;t exist</p>
        <p className={`${TYPOGRAPHY.body} text-muted-foreground max-w-sm`}>
          The page you&apos;re looking for isn&apos;t part of this workspace.
        </p>
        <Button nativeButton={false} render={<Link href="/" />}>
          Return to your documents
        </Button>
      </div>
    </AppShell>
  );
}
