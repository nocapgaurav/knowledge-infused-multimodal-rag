"use client";

import { FileSearch } from "lucide-react";

import { EvidenceCard } from "@/components/evidence/evidence-card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TYPOGRAPHY } from "@/constants/typography";
import { EMPTY_TURNS, useConversationStore } from "@/store/conversation-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import type { EvidenceItem } from "@/types/view-models";

/** Evidence is more important than generated text (Phase 3): its own
 * panel, independent of the conversation, never "hidden metadata." */
export function EvidenceList({ documentId }: { documentId: string }) {
  const turns = useConversationStore((state) => state.turnsByDocument[documentId] ?? EMPTY_TURNS);
  const openedEvidenceId = useWorkspaceStore((state) => state.openedEvidenceId);
  const openEvidence = useWorkspaceStore((state) => state.openEvidence);

  const items = dedupeEvidence(
    turns
      .filter((turn) => turn.status === "complete" && turn.answer)
      .flatMap((turn) => turn.answer!.evidenceItems)
      .reverse(),
  );

  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-12 text-center">
        <FileSearch className="text-muted-foreground size-6" aria-hidden="true" />
        <p className={TYPOGRAPHY.body}>Evidence appears here once you ask a question.</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-2 p-3">
        {items.map((item) => (
          <EvidenceCard
            key={item.knowledgeUnitId}
            item={item}
            isActive={item.knowledgeUnitId === openedEvidenceId}
            onSelect={() => openEvidence(item.knowledgeUnitId)}
          />
        ))}
      </div>
    </ScrollArea>
  );
}

function dedupeEvidence(items: EvidenceItem[]): EvidenceItem[] {
  const seen = new Set<string>();
  const result: EvidenceItem[] = [];
  for (const item of items) {
    if (seen.has(item.knowledgeUnitId)) continue;
    seen.add(item.knowledgeUnitId);
    result.push(item);
  }
  return result;
}
