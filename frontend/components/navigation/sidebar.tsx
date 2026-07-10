"use client";

import Link from "next/link";
import { useState } from "react";
import { FileText, Loader2, Trash2, Upload, XCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
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
  const selectedDocumentId = useWorkspaceStore((state) => state.selectedDocumentId);
  const selectDocument = useWorkspaceStore((state) => state.selectDocument);

  const documentList = Object.values(documents)
    .filter((doc) => doc.filename.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt));

  return (
    <aside className="flex h-full w-full flex-col gap-3 border-r p-3">
      <Button nativeButton={false} render={<Link href="/" />} className="w-full justify-start">
        <Upload className="size-4" />
        Upload paper
      </Button>

      {documentList.length > 0 && (
        <Input
          name="document-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search your documents"
          aria-label="Search your documents"
        />
      )}

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-1">
          {documentList.length === 0 ? (
            <p className={cn(TYPOGRAPHY.caption, "px-2 py-4 text-center")}>
              {query
                ? "No documents match your search."
                : "No documents yet. Upload a paper to get started."}
            </p>
          ) : (
            documentList.map((doc) => (
              <DocumentListItem
                key={doc.documentId}
                document={doc}
                isActive={doc.documentId === selectedDocumentId}
                onSelect={() => selectDocument(doc.documentId)}
                onRemove={() => removeDocument(doc.documentId)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </aside>
  );
}

function DocumentListItem({
  document,
  isActive,
  onSelect,
  onRemove,
}: {
  document: DocumentSummary;
  isActive: boolean;
  onSelect: () => void;
  onRemove: () => void;
}) {
  return (
    <Link
      href={document.status === "ready" ? `/workspace/${document.documentId}` : "/"}
      onClick={onSelect}
      className={cn(
        "group flex items-center gap-2 rounded-md px-2 py-2 text-left transition-colors",
        isActive ? "bg-accent text-accent-foreground" : "hover:bg-muted",
      )}
    >
      <StatusIcon status={document.status} />
      <span className={cn(TYPOGRAPHY.body, "min-w-0 flex-1 truncate")}>{document.filename}</span>
      <button
        type="button"
        aria-label={`Remove ${document.filename} from your list`}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onRemove();
        }}
        className="text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
      >
        <Trash2 className="size-3.5" />
      </button>
    </Link>
  );
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
