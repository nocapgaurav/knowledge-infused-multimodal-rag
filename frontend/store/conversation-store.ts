import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { ConversationTurn } from "@/types/view-models";

/**
 * A stable fallback for "this document has no turns yet." Selectors must
 * never fall back to a fresh `[]` literal -- Zustand's `useSyncExternalStore`
 * compares snapshots by reference, so a new array on every read (even one
 * that's semantically empty every time) looks like a perpetual state
 * change and triggers an infinite render loop ("Maximum update depth
 * exceeded"), confirmed via a real browser repro against the live backend.
 */
export const EMPTY_TURNS: readonly ConversationTurn[] = [];

/**
 * Conversation history, per document (Phase 4B: "Conversation history
 * should remain available throughout the session... Users should be
 * able to scroll naturally, resume reading, reference earlier answers").
 * Application state, not server state -- the backend has no notion of a
 * conversation; each turn is just one independent `/generate` call this
 * store threads into a history.
 */
interface ConversationState {
  turnsByDocument: Record<string, ConversationTurn[]>;

  startTurn: (documentId: string, turn: ConversationTurn) => void;
  completeTurn: (documentId: string, turnId: string, answer: ConversationTurn["answer"]) => void;
  failTurn: (documentId: string, turnId: string, reason: string) => void;
  clearConversation: (documentId: string) => void;
}

export const useConversationStore = create<ConversationState>()(
  persist(
    (set) => ({
      turnsByDocument: {},

      startTurn: (documentId, turn) =>
        set((state) => ({
          turnsByDocument: {
            ...state.turnsByDocument,
            [documentId]: [...(state.turnsByDocument[documentId] ?? []), turn],
          },
        })),

      completeTurn: (documentId, turnId, answer) =>
        set((state) => ({
          turnsByDocument: {
            ...state.turnsByDocument,
            [documentId]: (state.turnsByDocument[documentId] ?? []).map((turn) =>
              turn.id === turnId ? { ...turn, status: "complete", answer } : turn,
            ),
          },
        })),

      failTurn: (documentId, turnId, reason) =>
        set((state) => ({
          turnsByDocument: {
            ...state.turnsByDocument,
            [documentId]: (state.turnsByDocument[documentId] ?? []).map((turn) =>
              turn.id === turnId ? { ...turn, status: "failed", failureReason: reason } : turn,
            ),
          },
        })),

      clearConversation: (documentId) =>
        set((state) => ({
          turnsByDocument: { ...state.turnsByDocument, [documentId]: [] },
        })),
    }),
    { name: "conversation-store" },
  ),
);
