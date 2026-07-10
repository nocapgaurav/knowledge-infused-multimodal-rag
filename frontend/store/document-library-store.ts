import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { PreparationStage } from "@/services/document-service";
import type { DocumentDisplayStatus, DocumentSummary } from "@/types/view-models";

/**
 * The client-side document library (Phase 4A "Document List" / "Recent
 * Documents"). Modules 1-11 expose no list, search, or delete endpoint --
 * see the module12-backend-integration-gaps memory -- so this store,
 * not the backend, is the sole source of truth for "which documents has
 * this browser uploaded." Persisted so it survives a reload.
 */
interface DocumentRecord extends DocumentSummary {
  /** Internal-only bookkeeping for progress reporting; never rendered
   * verbatim (Phase 4A: "Never expose backend stage names"). */
  pipelineStage?: PreparationStage;
}

interface DocumentLibraryState {
  documents: Record<string, DocumentRecord>;

  addDocument: (documentId: string, filename: string) => void;
  setPipelineStage: (documentId: string, stage: PreparationStage) => void;
  setStatus: (documentId: string, status: DocumentDisplayStatus, failureReason?: string) => void;
  /** "Delete" only ever removes the local entry -- there is no backend
   * deletion capability to call. */
  removeDocument: (documentId: string) => void;
}

export const useDocumentLibraryStore = create<DocumentLibraryState>()(
  persist(
    (set) => ({
      documents: {},

      addDocument: (documentId, filename) =>
        set((state) => ({
          documents: {
            ...state.documents,
            [documentId]: {
              documentId,
              filename,
              uploadedAt: new Date().toISOString(),
              status: "preparing",
            },
          },
        })),

      setPipelineStage: (documentId, stage) =>
        set((state) => {
          const existing = state.documents[documentId];
          if (!existing) return state;
          return {
            documents: { ...state.documents, [documentId]: { ...existing, pipelineStage: stage } },
          };
        }),

      setStatus: (documentId, status, failureReason) =>
        set((state) => {
          const existing = state.documents[documentId];
          if (!existing) return state;
          return {
            documents: {
              ...state.documents,
              [documentId]: { ...existing, status, failureReason },
            },
          };
        }),

      removeDocument: (documentId) =>
        set((state) => {
          const rest = { ...state.documents };
          delete rest[documentId];
          return { documents: rest };
        }),
    }),
    { name: "document-library-store" },
  ),
);
