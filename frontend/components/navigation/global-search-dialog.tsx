"use client";

import { useRouter } from "next/navigation";

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { useDocumentLibraryStore } from "@/store/document-library-store";
import { useWorkspaceStore } from "@/store/workspace-store";

/**
 * Global search (Phase 4C): local-only, scoped to this browser's own
 * documents -- never the internet, never fabricated results. Reachable
 * from anywhere via the Top Navigation's Search button or Cmd/Ctrl+K.
 */
export function GlobalSearchDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const documents = useDocumentLibraryStore((state) => state.documents);
  const selectDocument = useWorkspaceStore((state) => state.selectDocument);

  const readyDocuments = Object.values(documents).filter((doc) => doc.status === "ready");

  function goToDocument(documentId: string) {
    selectDocument(documentId);
    onOpenChange(false);
    router.push(`/workspace/${documentId}`);
  }

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title="Search documents">
      <Command>
        <CommandInput placeholder="Search your documents..." />
        <CommandList>
          <CommandEmpty>No documents match your search.</CommandEmpty>
          <CommandGroup heading="Documents">
            {readyDocuments.map((doc) => (
              <CommandItem
                key={doc.documentId}
                value={doc.filename}
                onSelect={() => goToDocument(doc.documentId)}
              >
                {doc.filename}
              </CommandItem>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  );
}
