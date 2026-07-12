"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FlaskConical, Search, Settings } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/settings/theme-toggle";
import { TYPOGRAPHY } from "@/constants/typography";
import { useDocumentLibraryStore } from "@/store/document-library-store";
import { useWorkspaceStore } from "@/store/workspace-store";

/**
 * Global application controls (Phase 4A). Kept lightweight and consuming
 * minimal vertical space -- it never grows to carry workspace-specific
 * content, that belongs to the panels beneath it.
 */
export function TopNav({ onOpenSearch }: { onOpenSearch: () => void }) {
  const pathname = usePathname();
  const selectedDocumentId = useWorkspaceStore((state) => state.selectedDocumentId);
  const currentDocument = useDocumentLibraryStore((state) =>
    selectedDocumentId ? state.documents[selectedDocumentId] : undefined,
  );
  // selectedDocumentId is persisted and only ever set, never cleared, on
  // navigating away from a workspace route -- so the title must be gated
  // on actually being on that document's route, not just on the store.
  const isWorkspaceRoute = pathname?.startsWith("/workspace/") ?? false;

  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b px-4">
      <Link href="/" className="flex items-center gap-2">
        <FlaskConical className="text-primary size-5" aria-hidden="true" />
        <span className={TYPOGRAPHY.appTitle}>Research Workspace</span>
      </Link>

      <div className={`min-w-0 flex-1 text-center ${TYPOGRAPHY.workspaceTitle} truncate`}>
        {isWorkspaceRoute && currentDocument ? currentDocument.filename : ""}
      </div>

      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" onClick={onOpenSearch} aria-label="Search documents">
          <Search className="size-4" />
          <span className="hidden sm:inline">Search</span>
          <kbd className="text-muted-foreground bg-muted ml-1 hidden rounded px-1.5 py-0.5 text-xs sm:inline">
            ⌘K
          </kbd>
        </Button>
        <ThemeToggle />
        <Button
          variant="ghost"
          size="icon"
          aria-label="Settings"
          nativeButton={false}
          render={<Link href="/settings" />}
        >
          <Settings className="size-4" />
        </Button>
      </div>
    </header>
  );
}
