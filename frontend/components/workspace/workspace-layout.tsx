"use client";

import { ErrorBoundary } from "@/components/feedback/error-boundary";
import { ConversationPanel } from "@/components/conversation/conversation-panel";
import { EvidencePdfPanel } from "@/components/workspace/evidence-pdf-panel";
import { PANEL_WIDTH } from "@/constants/layout";
import { useWorkspaceStore } from "@/store/workspace-store";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import type { Layout } from "react-resizable-panels";

const CONVERSATION_PANEL_ID = "conversation";

/**
 * The research workspace (Phase 3/4A): Conversation (left), Evidence/PDF
 * (right, adaptive) -- the former third "related content" panel was
 * removed for not earning its screen space, and the PDF/Evidence panel
 * simply grows to fill it. Each panel stays independently error-isolated
 * (Phase 4D: "Failure inside one panel must never crash another") and a
 * future panel can be reintroduced the same way: a new `ResizablePanel`
 * plus `ResizableHandle`, its own store-backed size, and its own
 * `ErrorBoundary` -- nothing else here would need to change.
 */
export function WorkspaceLayout({ documentId }: { documentId: string }) {
  const conversationPercent = useWorkspaceStore((state) => state.conversationPanelPercent);
  const setPanelSizes = useWorkspaceStore((state) => state.setPanelSizes);

  function handleLayoutChanged(layout: Layout) {
    const conversation = layout[CONVERSATION_PANEL_ID];
    if (conversation !== undefined) {
      setPanelSizes(conversation);
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
        defaultSize={100 - conversationPercent}
        minSize={PANEL_WIDTH.panelMinPercent}
      >
        <ErrorBoundary label="Evidence and PDF">
          <EvidencePdfPanel documentId={documentId} />
        </ErrorBoundary>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
