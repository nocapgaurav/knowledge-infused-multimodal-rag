"use client";

import { ErrorBoundary } from "@/components/feedback/error-boundary";
import { ConversationPanel } from "@/components/conversation/conversation-panel";
import { EvidencePdfPanel } from "@/components/workspace/evidence-pdf-panel";
import { RelatedContentPanel } from "@/components/related/related-content-panel";
import { PANEL_WIDTH } from "@/constants/layout";
import { useWorkspaceStore } from "@/store/workspace-store";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import type { Layout } from "react-resizable-panels";

const CONVERSATION_PANEL_ID = "conversation";
const CONTEXT_PANEL_ID = "context";

/**
 * The 3-panel research workspace (Phase 3/4A): Conversation (left),
 * Evidence/PDF (center, adaptive), Related Content/Context (right).
 * Each panel is independently error-isolated (Phase 4D: "Failure inside
 * one panel must never crash another") and resizes/persists via the
 * workspace store.
 */
export function WorkspaceLayout({ documentId }: { documentId: string }) {
  const conversationPercent = useWorkspaceStore((state) => state.conversationPanelPercent);
  const contextPercent = useWorkspaceStore((state) => state.contextPanelPercent);
  const setPanelSizes = useWorkspaceStore((state) => state.setPanelSizes);

  function handleLayoutChanged(layout: Layout) {
    const conversation = layout[CONVERSATION_PANEL_ID];
    const context = layout[CONTEXT_PANEL_ID];
    if (conversation !== undefined && context !== undefined) {
      setPanelSizes(conversation, context);
    }
  }

  return (
    <ResizablePanelGroup orientation="horizontal" onLayoutChanged={handleLayoutChanged}>
      <ResizablePanel
        id={CONVERSATION_PANEL_ID}
        defaultSize={conversationPercent}
        minSize={PANEL_WIDTH.panelMinPercent}
      >
        <ErrorBoundary label="Conversation">
          <ConversationPanel documentId={documentId} />
        </ErrorBoundary>
      </ResizablePanel>

      <ResizableHandle withHandle />

      <ResizablePanel
        id="evidence-pdf"
        defaultSize={100 - conversationPercent - contextPercent}
        minSize={PANEL_WIDTH.panelMinPercent}
      >
        <ErrorBoundary label="Evidence and PDF">
          <EvidencePdfPanel documentId={documentId} />
        </ErrorBoundary>
      </ResizablePanel>

      <ResizableHandle withHandle />

      <ResizablePanel
        id={CONTEXT_PANEL_ID}
        defaultSize={contextPercent}
        minSize={PANEL_WIDTH.panelMinPercent}
      >
        <ErrorBoundary label="Related content">
          <RelatedContentPanel documentId={documentId} />
        </ErrorBoundary>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
