"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowDown, Loader2, MessageSquare, Trash2 } from "lucide-react";
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
import type { Citation, ConversationTurn } from "@/types/view-models";

/** Honest, reader-facing phrasing for each evidence status. In particular
 * "unverified_answer" must not read as "the paper lacks this" -- evidence
 * was found; the answer's citations just could not be verified. */
const STATUS_BADGE_TEXT = {
  sufficient_evidence: "Fully grounded in evidence",
  partially_sufficient_evidence: "Partially verified against evidence",
  unverified_answer: "Citations not verified",
  insufficient_evidence: "Not found in this paper",
} as const;

/** How close to the bottom (in pixels) still counts as "at the bottom" --
 * generous enough that the tail end of a message doesn't register as a
 * deliberate scroll-away. */
const BOTTOM_THRESHOLD_PX = 96;

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

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  // A ref mirrors the state so the ResizeObserver callback (created once)
  // always reads the latest value instead of one captured at mount.
  const isAtBottomRef = useRef(true);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const scrollToBottom = useCallback((behavior: ScrollBehavior) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.scrollTo({ top: viewport.scrollHeight, behavior });
  }, []);

  const handleViewportScroll = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const distanceFromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    const atBottom = distanceFromBottom <= BOTTOM_THRESHOLD_PX;
    isAtBottomRef.current = atBottom;
    setIsAtBottom(atBottom);
  }, []);

  // Follow new content -- a turn being appended, an answer replacing its
  // "thinking" placeholder -- only while the reader hasn't deliberately
  // scrolled away. A reader mid-scroll-up must never be yanked back down.
  useEffect(() => {
    const content = contentRef.current;
    if (!content) return;
    const observer = new ResizeObserver(() => {
      if (isAtBottomRef.current) scrollToBottom("auto");
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [scrollToBottom]);

  function handleClear() {
    clearConversation(documentId);
    toast.success("Conversation cleared");
  }

  function handleSubmit(question: string) {
    // Sending a question always returns the reader to the live edge of the
    // conversation, even if they had scrolled up to reread something.
    isAtBottomRef.current = true;
    setIsAtBottom(true);
    ask.mutate(question);
    requestAnimationFrame(() => scrollToBottom("smooth"));
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
      <div className="relative min-h-0 flex-1">
        <ScrollArea
          className="h-full"
          viewportRef={viewportRef}
          onViewportScroll={handleViewportScroll}
        >
          <div ref={contentRef} className="flex flex-col gap-6 p-4">
            {turns.length === 0 ? (
              <EmptyConversation />
            ) : (
              turns.map((turn) => (
                <ConversationTurnView
                  key={turn.id}
                  turn={turn}
                  onSelectCitation={(citation) =>
                    openEvidence(citation.knowledgeUnitId, {
                      text: citation.textExcerpt,
                      displayLabel: citation.displayLabel,
                      pageNumbers: citation.pageNumbers,
                      boundingBoxes: citation.boundingBoxes,
                      modality: citation.modality,
                    })
                  }
                />
              ))
            )}
          </div>
        </ScrollArea>
        <ScrollToLatestButton
          visible={!isAtBottom && turns.length > 0}
          onClick={() => scrollToBottom("smooth")}
        />
      </div>
      <QuestionInput onSubmit={handleSubmit} isSubmitting={ask.isPending} />
    </div>
  );
}

function ScrollToLatestButton({ visible, onClick }: { visible: boolean; onClick: () => void }) {
  return (
    <Button
      type="button"
      variant="secondary"
      size="sm"
      onClick={onClick}
      aria-label="Scroll to latest message"
      tabIndex={visible ? 0 : -1}
      className={cn(
        "absolute bottom-3 left-1/2 -translate-x-1/2 gap-1.5 rounded-full border shadow-md transition-all duration-200 ease-out",
        visible ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0",
      )}
    >
      <ArrowDown className="size-3.5" />
      Scroll to latest
    </Button>
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
  onSelectCitation: (citation: Citation) => void;
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
        <div className={cn(TYPOGRAPHY.caption, "flex items-center gap-1.5")}>
          <Loader2 className="size-3 animate-spin" aria-hidden="true" />
          <span className="animate-pulse">Thinking through the evidence...</span>
        </div>
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
