"use client";

import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";

import { generateAnswer } from "@/services/generation-service";
import { useConversationStore } from "@/store/conversation-store";
import { AppError } from "@/types/errors";

/** The Question Journey (Phase 4D): entered -> validated -> submitted ->
 * loading feedback -> grounded answer -> conversation updated. Duplicate
 * submissions are prevented by the mutation's own pending state, checked
 * by the input component before calling `ask`. */
export function useAskQuestion(documentId: string) {
  const startTurn = useConversationStore((state) => state.startTurn);
  const completeTurn = useConversationStore((state) => state.completeTurn);
  const failTurn = useConversationStore((state) => state.failTurn);

  const mutation = useMutation({
    mutationFn: async (question: string) => {
      const turnId = crypto.randomUUID();
      startTurn(documentId, {
        id: turnId,
        question,
        askedAt: new Date().toISOString(),
        status: "pending",
      });

      try {
        if (!navigator.onLine) {
          throw new AppError("You're offline. Reconnect to ask a question.", "network");
        }
        const { answer } = await generateAnswer(documentId, question);
        completeTurn(documentId, turnId, answer);
        return answer;
      } catch (error) {
        const message = error instanceof AppError ? error.message : "Could not generate an answer.";
        failTurn(documentId, turnId, message);
        toast.error("Couldn't answer that question", { description: message });
        throw error;
      }
    },
  });

  return mutation;
}
