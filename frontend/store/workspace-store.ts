import { create } from "zustand";
import { persist } from "zustand/middleware";

import { PANEL_WIDTH } from "@/constants/layout";

/**
 * The sole owner of application/UI state (Phase 2B). Never server state
 * (that's TanStack Query's job -- see `services/`) and never anything
 * that could instead be derived. Persisted so a returning user's
 * workspace, panel state, and selection are restored without rebuilding
 * (Phase 4C/4D: Session Persistence, Workspace Memory).
 */
interface WorkspaceState {
  selectedDocumentId: string | null;
  openedEvidenceId: string | null;
  sidebarOpen: boolean;
  contextPanelOpen: boolean;
  conversationPanelPercent: number;
  contextPanelPercent: number;
  lastPdfPageByDocument: Record<string, number>;

  selectDocument: (documentId: string | null) => void;
  openEvidence: (knowledgeUnitId: string | null) => void;
  toggleSidebar: () => void;
  toggleContextPanel: () => void;
  setPanelSizes: (conversationPercent: number, contextPercent: number) => void;
  setLastPdfPage: (documentId: string, page: number) => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      selectedDocumentId: null,
      openedEvidenceId: null,
      sidebarOpen: true,
      contextPanelOpen: true,
      conversationPanelPercent: PANEL_WIDTH.conversationDefaultPercent,
      contextPanelPercent: PANEL_WIDTH.contextDefaultPercent,
      lastPdfPageByDocument: {},

      selectDocument: (documentId) =>
        set({ selectedDocumentId: documentId, openedEvidenceId: null }),
      openEvidence: (knowledgeUnitId) => set({ openedEvidenceId: knowledgeUnitId }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      toggleContextPanel: () => set((state) => ({ contextPanelOpen: !state.contextPanelOpen })),
      setPanelSizes: (conversationPercent, contextPercent) =>
        set({ conversationPanelPercent: conversationPercent, contextPanelPercent: contextPercent }),
      setLastPdfPage: (documentId, page) =>
        set((state) => ({
          lastPdfPageByDocument: { ...state.lastPdfPageByDocument, [documentId]: page },
        })),
    }),
    { name: "workspace-store" },
  ),
);
