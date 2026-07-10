"use client";

import { useEffect } from "react";
import { useParams } from "next/navigation";
import { FileWarning, Loader2 } from "lucide-react";

import { WorkspaceLayout } from "@/components/workspace/workspace-layout";
import { TYPOGRAPHY } from "@/constants/typography";
import { useDocumentLibraryStore } from "@/store/document-library-store";
import { useWorkspaceStore } from "@/store/workspace-store";

/**
 * The Research Workspace (Phase 4A/4B): the product's primary screen.
 * `documentId` is real URL state (Phase 2B) so a workspace is
 * deep-linkable and refresh-safe, not just held in memory.
 */
export default function WorkspacePage() {
  const { documentId } = useParams<{ documentId: string }>();
  const document = useDocumentLibraryStore((state) => state.documents[documentId]);
  const selectDocument = useWorkspaceStore((state) => state.selectDocument);

  useEffect(() => {
    selectDocument(documentId);
  }, [documentId, selectDocument]);

  if (!document) {
    return (
      <EmptyWorkspaceState
        icon={<FileWarning className="text-muted-foreground size-8" aria-hidden="true" />}
        title="This document isn't in your library"
        description="It may have been uploaded from a different browser, or removed from your recent documents. Upload it again to continue."
      />
    );
  }

  if (document.status === "preparing") {
    return (
      <EmptyWorkspaceState
        icon={<Loader2 className="text-primary size-8 animate-spin" aria-hidden="true" />}
        title="Preparing document..."
        description="This usually takes under a minute."
      />
    );
  }

  if (document.status === "failed") {
    return (
      <EmptyWorkspaceState
        icon={<FileWarning className="text-error size-8" aria-hidden="true" />}
        title="This document couldn't be prepared"
        description={
          document.failureReason ?? "Something went wrong while preparing this document."
        }
      />
    );
  }

  return <WorkspaceLayout documentId={documentId} />;
}

function EmptyWorkspaceState({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
      {icon}
      <p className={TYPOGRAPHY.workspaceTitle}>{title}</p>
      <p className={`${TYPOGRAPHY.body} text-muted-foreground max-w-sm`}>{description}</p>
    </div>
  );
}
