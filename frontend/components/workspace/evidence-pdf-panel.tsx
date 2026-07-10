"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { EvidenceList } from "@/components/evidence/evidence-list";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EMPTY_TURNS, useConversationStore } from "@/store/conversation-store";
import { useWorkspaceStore } from "@/store/workspace-store";

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
 * Evidence / PDF (adaptive)"). Selecting evidence switches to the PDF
 * tab automatically so the sync feels immediate. */
export function EvidencePdfPanel({ documentId }: { documentId: string }) {
  const openedEvidenceId = useWorkspaceStore((state) => state.openedEvidenceId);
  const turns = useConversationStore((state) => state.turnsByDocument[documentId] ?? EMPTY_TURNS);
  const evidenceText = turns
    .filter((turn) => turn.status === "complete" && turn.answer)
    .flatMap((turn) => turn.answer!.evidenceItems)
    .find((item) => item.knowledgeUnitId === openedEvidenceId)?.text;

  const [activeTab, setActiveTab] = useState<"evidence" | "pdf">("evidence");

  useEffect(() => {
    if (openedEvidenceId) setActiveTab("pdf");
  }, [openedEvidenceId]);

  return (
    <Tabs
      value={activeTab}
      onValueChange={(value) => setActiveTab(value as "evidence" | "pdf")}
      className="flex h-full flex-col gap-0"
    >
      <TabsList className="m-2 w-fit">
        <TabsTrigger value="evidence">Evidence</TabsTrigger>
        <TabsTrigger value="pdf">PDF</TabsTrigger>
      </TabsList>
      <TabsContent value="evidence" className="min-h-0 flex-1">
        <EvidenceList documentId={documentId} />
      </TabsContent>
      <TabsContent value="pdf" className="min-h-0 flex-1">
        <PdfViewer documentId={documentId} searchText={evidenceText ?? null} />
      </TabsContent>
    </Tabs>
  );
}
