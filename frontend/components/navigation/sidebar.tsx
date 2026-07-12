"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  FileSearch,
  FileText,
  Loader2,
  MoreVertical,
  Pencil,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TYPOGRAPHY } from "@/constants/typography";
import { cn } from "@/lib/utils";
import { useDocumentLibraryStore } from "@/store/document-library-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import type { DocumentSummary } from "@/types/view-models";

/**
 * Document management, and nothing else (Phase 4A: "Purpose: Document
 * management. Nothing else."). The list itself is client-owned -- see
 * module12-backend-integration-gaps memory -- refreshed against the real
 * per-document status endpoint elsewhere (Task #75's upload flow), not
 * queried from a backend "list" that doesn't exist.
 */
export function Sidebar() {
  const [query, setQuery] = useState("");
  const documents = useDocumentLibraryStore((state) => state.documents);
  const removeDocument = useDocumentLibraryStore((state) => state.removeDocument);
  const renameDocument = useDocumentLibraryStore((state) => state.renameDocument);
  const selectedDocumentId = useWorkspaceStore((state) => state.selectedDocumentId);
  const selectDocument = useWorkspaceStore((state) => state.selectDocument);

  const allDocuments = Object.values(documents);
  const documentList = allDocuments
    .filter((doc) => doc.filename.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt));

  return (
    <aside className="flex h-full w-full flex-col gap-4 border-r p-3">
      <Button nativeButton={false} render={<Link href="/" />} className="w-full justify-start">
        <Upload className="size-4" />
        Upload paper
      </Button>

      {allDocuments.length > 0 && (
        <Input
          name="document-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search your documents"
          aria-label="Search your documents"
        />
      )}

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-1.5">
          {documentList.length === 0 ? (
            <EmptyDocumentList isSearching={query.length > 0} />
          ) : (
            documentList.map((doc) => (
              <DocumentListItem
                key={doc.documentId}
                document={doc}
                isActive={doc.documentId === selectedDocumentId}
                onSelect={() => selectDocument(doc.documentId)}
                onRemove={() => removeDocument(doc.documentId)}
                onRename={(filename) => renameDocument(doc.documentId, filename)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </aside>
  );
}

function EmptyDocumentList({ isSearching }: { isSearching: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2 px-2 py-10 text-center">
      <FileSearch className="text-muted-foreground/60 size-6" aria-hidden="true" />
      {isSearching ? (
        <>
          <p className={TYPOGRAPHY.body}>No papers found</p>
          <p className={TYPOGRAPHY.caption}>Try another keyword.</p>
        </>
      ) : (
        <>
          <p className={TYPOGRAPHY.body}>No documents yet</p>
          <p className={TYPOGRAPHY.caption}>Upload a paper to get started.</p>
        </>
      )}
    </div>
  );
}

function DocumentListItem({
  document,
  isActive,
  onSelect,
  onRemove,
  onRename,
}: {
  document: DocumentSummary;
  isActive: boolean;
  onSelect: () => void;
  onRemove: () => void;
  onRename: (filename: string) => void;
}) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [draftName, setDraftName] = useState(document.filename);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (isRenaming) inputRef.current?.select();
  }, [isRenaming]);

  function startRename() {
    setDraftName(document.filename);
    setIsRenaming(true);
  }

  function commitRename() {
    const trimmed = draftName.trim();
    if (trimmed && trimmed !== document.filename) onRename(trimmed);
    setIsRenaming(false);
  }

  // The active card is meant to be unmistakable at a glance, but lightly
  // so: a thin left accent bar, a slightly brighter surface, a thin
  // border -- no badge, no heavy elevation. All colors come from theme
  // tokens so light, dark, and high-contrast modes stay consistent
  // without a separate palette.
  const cardClassName = cn(
    "group relative flex items-start gap-2 rounded-md border py-2 pr-1 pl-3 text-left transition-all duration-200",
    isActive
      ? "bg-accent text-accent-foreground border-border before:absolute before:inset-y-1.5 before:left-0.5 before:w-[3px] before:rounded-full before:bg-primary"
      : "border-transparent hover:bg-muted",
  );
  const { stem, extension } = splitFilename(document.filename);

  if (isRenaming) {
    return (
      <div className={cardClassName}>
        <StatusIcon status={document.status} />
        <input
          ref={inputRef}
          name="rename-document"
          value={draftName}
          onChange={(event) => setDraftName(event.target.value)}
          onBlur={commitRename}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commitRename();
            }
            if (event.key === "Escape") {
              event.preventDefault();
              setIsRenaming(false);
            }
          }}
          autoFocus
          aria-label={`Rename ${document.filename}`}
          className={cn(
            TYPOGRAPHY.body,
            "border-ring bg-background min-w-0 flex-1 rounded border px-1 py-0.5 outline-none",
          )}
        />
      </div>
    );
  }

  return (
    <Link
      href={document.status === "ready" ? `/workspace/${document.documentId}` : "/"}
      onClick={onSelect}
      className={cardClassName}
    >
      <StatusIcon status={document.status} />
      <div
        className="flex min-w-0 flex-1 items-baseline"
        title={document.filename}
      >
        <span
          className={cn(TYPOGRAPHY.body, "min-w-0 truncate", isActive && "font-medium")}
        >
          {stem}
        </span>
        {extension && (
          <span className={cn(TYPOGRAPHY.body, "shrink-0", isActive && "font-medium")}>
            {extension}
          </span>
        )}
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <button
              type="button"
              aria-label={`More actions for ${document.filename}`}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
              }}
              className={cn(
                "text-muted-foreground hover:text-foreground hover:bg-background/60 data-open:opacity-100 data-open:bg-background/60 shrink-0 self-start rounded p-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100 focus-visible:opacity-100",
                isActive && "opacity-100",
              )}
            >
              <MoreVertical className="size-3.5" />
            </button>
          }
        />
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onClick={(event) => {
              // Base UI portals this menu outside the Link's DOM subtree,
              // but React still bubbles the click through the component
              // tree -- without stopping it here, it reaches the Link's
              // onClick and silently re-selects whichever row's menu was
              // used, regardless of which document is actually open.
              event.preventDefault();
              event.stopPropagation();
              startRename();
            }}
          >
            <Pencil />
            Rename
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            variant="destructive"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onRemove();
            }}
          >
            <Trash2 />
            Remove
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </Link>
  );
}

/** Splits a filename into a truncatable stem and its extension, purely
 * for layout -- `stem + extension` always reconstructs the exact
 * original string. The extension renders in its own non-truncating span
 * so it stays visible even when the stem is cut off with an ellipsis. */
function splitFilename(filename: string): { stem: string; extension: string } {
  const lastDot = filename.lastIndexOf(".");
  if (lastDot <= 0 || lastDot === filename.length - 1) {
    return { stem: filename, extension: "" };
  }
  return { stem: filename.slice(0, lastDot), extension: filename.slice(lastDot) };
}

function StatusIcon({ status }: { status: DocumentSummary["status"] }) {
  if (status === "preparing") {
    return (
      <Loader2
        className="text-muted-foreground size-4 shrink-0 animate-spin"
        aria-label="Preparing"
      />
    );
  }
  if (status === "failed") {
    return <XCircle className="text-error size-4 shrink-0" aria-label="Failed" />;
  }
  return <FileText className="text-muted-foreground size-4 shrink-0" aria-label="Ready" />;
}
