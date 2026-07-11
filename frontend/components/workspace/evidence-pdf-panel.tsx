"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Loader2 } from "lucide-react";

import { EvidenceList } from "@/components/evidence/evidence-list";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TYPOGRAPHY } from "@/constants/typography";
import { EMPTY_TURNS, useConversationStore } from "@/store/conversation-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import type { EvidenceItem, EvidenceTarget } from "@/types/view-models";

const PdfViewer = dynamic(
  () => import("@/components/pdf/pdf-viewer").then((module) => module.PdfViewer),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="text-muted-foreground size-6 animate-spin" />
      </div>
    ),
  },
);

/** Evidence and PDF share the center panel via tabs (Phase 3: "Center:
 * Evidence / PDF (adaptive)"). Selecting evidence anywhere switches to
 * the PDF tab automatically so the sync feels immediate; the evidence
 * bar lets a reader step through every location supporting the current
 * answer without returning to the conversation. */
export function EvidencePdfPanel({ documentId }: { documentId: string }) {
  const openedEvidenceId = useWorkspaceStore((state) => state.openedEvidenceId);
  const openedEvidenceTarget = useWorkspaceStore((state) => state.openedEvidenceTarget);
  const openEvidence = useWorkspaceStore((state) => state.openEvidence);
  const turns = useConversationStore((state) => state.turnsByDocument[documentId] ?? EMPTY_TURNS);

  // The navigable evidence set is scoped to ONE answer: the turn that
  // contains the currently opened evidence (falling back to the latest
  // answer). Aggregating across turns produced ambiguous indices --
  // citation labels repeat per answer (turn 1's KU4 and turn 3's KU4 are
  // different chunks), and prev/next must never wander into another
  // answer's evidence.
  const completeTurns = turns.filter((turn) => turn.status === "complete" && turn.answer);
  const owningTurn =
    (openedEvidenceId
      ? [...completeTurns]
          .reverse()
          .find((turn) =>
            turn.answer!.evidenceItems.some((item) => item.knowledgeUnitId === openedEvidenceId),
          )
      : undefined) ?? completeTurns[completeTurns.length - 1];
  const answerEvidence: EvidenceItem[] = owningTurn?.answer?.evidenceItems ?? [];

  // The canonical selection is the id; the target rides along from the
  // click. After a refresh (persisted id, possibly stale target) the
  // conversation store remains the fallback source.
  const fallbackItem = answerEvidence.find((item) => item.knowledgeUnitId === openedEvidenceId);
  const target: EvidenceTarget | null =
    openedEvidenceTarget ??
    (fallbackItem
      ? {
          text: fallbackItem.text,
          displayLabel: fallbackItem.displayLabel,
          pageNumbers: fallbackItem.pageNumbers,
          boundingBoxes: fallbackItem.boundingBoxes,
        }
      : null);

  const [activeTab, setActiveTab] = useState<"evidence" | "pdf">("evidence");

  useEffect(() => {
    if (openedEvidenceId) setActiveTab("pdf");
  }, [openedEvidenceId]);

  const currentIndex = answerEvidence.findIndex(
    (item) => item.knowledgeUnitId === openedEvidenceId,
  );

  function step(offset: number) {
    if (!answerEvidence.length) return;
    const next =
      currentIndex === -1
        ? 0
        : (currentIndex + offset + answerEvidence.length) % answerEvidence.length;
    const item = answerEvidence[next];
    if (!item) return;
    openEvidence(item.knowledgeUnitId, {
      text: item.text,
      displayLabel: item.displayLabel,
      pageNumbers: item.pageNumbers,
      boundingBoxes: item.boundingBoxes,
    });
  }

  return (
    <Tabs
      value={activeTab}
      onValueChange={(value) => setActiveTab(value as "evidence" | "pdf")}
      className="flex h-full flex-col gap-0"
    >
      <div className="flex items-center gap-2 pr-2">
        <TabsList className="m-2 w-fit">
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="pdf">PDF</TabsTrigger>
        </TabsList>
        {activeTab === "pdf" && target && (
          <div className="ml-auto flex min-w-0 items-center gap-1">
            {answerEvidence.length > 1 && (
              <>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => step(-1)}
                  aria-label="Previous evidence"
                >
                  <ChevronLeft className="size-4" />
                </Button>
                <span className={TYPOGRAPHY.caption}>
                  {currentIndex === -1 ? "–" : currentIndex + 1}/{answerEvidence.length}
                </span>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => step(1)}
                  aria-label="Next evidence"
                >
                  <ChevronRight className="size-4" />
                </Button>
              </>
            )}
            <span className={`${TYPOGRAPHY.caption} min-w-0 truncate`}>
              Viewing: {target.displayLabel ?? "evidence"}
              {target.pageNumbers?.length ? ` · p. ${target.pageNumbers[0]}` : ""}
            </span>
          </div>
        )}
      </div>
      <TabsContent value="evidence" className="min-h-0 flex-1">
        <EvidenceList documentId={documentId} />
      </TabsContent>
      <TabsContent value="pdf" className="min-h-0 flex-1">
        <PdfViewer documentId={documentId} target={target} />
      </TabsContent>
    </Tabs>
  );
}
