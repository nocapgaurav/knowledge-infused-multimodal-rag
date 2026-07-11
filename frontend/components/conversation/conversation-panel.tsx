"use client";

import { motion, useReducedMotion } from "framer-motion";
import { MessageSquare, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { CitationText } from "@/components/conversation/citation-text";
import { QuestionInput } from "@/components/conversation/question-input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SLIDE_UP_VARIANTS } from "@/constants/motion";
import { TYPOGRAPHY } from "@/constants/typography";
import { useAskQuestion } from "@/hooks/use-ask-question";
import { cn } from "@/lib/utils";
import { useAccessibilityStore } from "@/store/accessibility-store";
import { EMPTY_TURNS, useConversationStore } from "@/store/conversation-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import type { ConversationTurn } from "@/types/view-models";

/** Honest, reader-facing phrasing for each evidence status. In particular
 * "unverified_answer" must not read as "the paper lacks this" -- evidence
 * was found; the answer's citations just could not be verified. */
const STATUS_BADGE_TEXT = {
  sufficient_evidence: "Fully grounded in evidence",
  partially_sufficient_evidence: "Partially verified against evidence",
  unverified_answer: "Citations not verified",
  insufficient_evidence: "Not found in this paper",
} as const;

/**
 * The Conversation experience (Phase 4B): a tool for understanding the
 * paper, never the product itself -- every answer stays visibly tied to
 * its evidence via clickable citations.
 */
export function ConversationPanel({ documentId }: { documentId: string }) {
  const turns = useConversationStore((state) => state.turnsByDocument[documentId] ?? EMPTY_TURNS);
  const clearConversation = useConversationStore((state) => state.clearConversation);
  const ask = useAskQuestion(documentId);
  const openEvidence = useWorkspaceStore((state) => state.openEvidence);

  function handleClear() {
    clearConversation(documentId);
    toast.success("Conversation cleared");
  }

  return (
    <div className="flex h-full flex-col">
      {turns.length > 0 && (
        <div className="flex items-center justify-between border-b px-3 py-1.5">
          <span className={TYPOGRAPHY.caption}>Conversation</span>
          <Button variant="ghost" size="sm" onClick={handleClear} aria-label="Clear conversation">
            <Trash2 className="size-3.5" />
            Clear
          </Button>
        </div>
      )}
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-6 p-4">
          {turns.length === 0 ? (
            <EmptyConversation />
          ) : (
            turns.map((turn) => (
              <ConversationTurnView key={turn.id} turn={turn} onSelectCitation={openEvidence} />
            ))
          )}
        </div>
      </ScrollArea>
      <QuestionInput onSubmit={(question) => ask.mutate(question)} isSubmitting={ask.isPending} />
    </div>
  );
}

function EmptyConversation() {
  return (
    <div className="flex flex-col items-center gap-2 py-12 text-center">
      <MessageSquare className="text-muted-foreground size-6" aria-hidden="true" />
      <p className={TYPOGRAPHY.body}>Ask your first question about this paper.</p>
      <p className={TYPOGRAPHY.caption}>Every answer will show exactly where it came from.</p>
    </div>
  );
}

function ConversationTurnView({
  turn,
  onSelectCitation,
}: {
  turn: ConversationTurn;
  onSelectCitation: (knowledgeUnitId: string) => void;
}) {
  const explicitReducedMotion = useAccessibilityStore((state) => state.reducedMotion);
  const systemReducedMotion = useReducedMotion();
  const reduceMotion = explicitReducedMotion ?? systemReducedMotion ?? false;

  return (
    <motion.div
      initial={reduceMotion ? undefined : "hidden"}
      animate={reduceMotion ? undefined : "visible"}
      variants={reduceMotion ? undefined : SLIDE_UP_VARIANTS}
      className="flex flex-col gap-3"
    >
      <div className="flex justify-end">
        <div className="bg-conversation text-conversation-foreground max-w-[85%] rounded-lg px-3 py-2">
          <p className={TYPOGRAPHY.body}>{turn.question}</p>
        </div>
      </div>

      {turn.status === "pending" && (
        <p className={cn(TYPOGRAPHY.caption, "animate-pulse")}>Thinking through the evidence...</p>
      )}

      {turn.status === "failed" && (
        <p className="text-error text-sm">
          {turn.failureReason ?? "Could not answer this question."}
        </p>
      )}

      {turn.status === "complete" && turn.answer && (
        <div className="flex flex-col gap-2">
          <CitationText
            text={turn.answer.answer}
            citations={turn.answer.citations}
            onSelectCitation={onSelectCitation}
          />
          <Badge variant="outline" className="w-fit">
            {STATUS_BADGE_TEXT[turn.answer.answerStatus]}
          </Badge>
        </div>
      )}
    </motion.div>
  );
}
