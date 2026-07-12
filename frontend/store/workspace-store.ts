import { create } from "zustand";
import { persist } from "zustand/middleware";

import { PANEL_WIDTH } from "@/constants/layout";
import type { EvidenceTarget } from "@/types/view-models";

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
  /** The selection payload snapshotted when evidence was opened -- how
   * the PDF viewer locates the evidence regardless of which view
   * (citation, evidence card) initiated the selection.
   * `openedEvidenceId` remains the single synchronization key. */
  openedEvidenceTarget: EvidenceTarget | null;
  sidebarOpen: boolean;
  conversationPanelPercent: number;
  lastPdfPageByDocument: Record<string, number>;

  selectDocument: (documentId: string | null) => void;
  openEvidence: (knowledgeUnitId: string | null, target?: EvidenceTarget | null) => void;
  toggleSidebar: () => void;
  setPanelSizes: (conversationPercent: number) => void;
  setLastPdfPage: (documentId: string, page: number) => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      selectedDocumentId: null,
      openedEvidenceId: null,
      openedEvidenceTarget: null,
      sidebarOpen: true,
      conversationPanelPercent: PANEL_WIDTH.conversationDefaultPercent,
      lastPdfPageByDocument: {},

      selectDocument: (documentId) =>
        set({ selectedDocumentId: documentId, openedEvidenceId: null, openedEvidenceTarget: null }),
      openEvidence: (knowledgeUnitId, target) =>
        set({ openedEvidenceId: knowledgeUnitId, openedEvidenceTarget: target ?? null }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setPanelSizes: (conversationPercent) =>
        set({ conversationPanelPercent: conversationPercent }),
      setLastPdfPage: (documentId, page) =>
        set((state) => ({
          lastPdfPageByDocument: { ...state.lastPdfPageByDocument, [documentId]: page },
        })),
    }),
    {
      name: "workspace-store",
      // Bumped when the conversation/PDF default changed to an even
      // 50/50 split: a returning browser's already-persisted
      // `conversationPanelPercent` (saved under the old default) must
      // not silently keep overriding the new one forever. Dropping the
      // key here -- rather than migrating its value -- lets the fresh
      // default from the store's initial state take over exactly once;
      // a genuine resize made after this point persists normally, same
      // as before.
      version: 1,
      migrate: (persistedState, version) => {
        const state = { ...(persistedState as Record<string, unknown>) };
        if (version < 1) {
          delete state.conversationPanelPercent;
          // Dead keys from the now-removed related-content panel.
          delete state.contextPanelPercent;
          delete state.contextPanelOpen;
        }
        return state;
      },
    },
  ),
);
