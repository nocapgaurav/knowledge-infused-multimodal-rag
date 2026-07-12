"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Loader2, Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export function QuestionInput({
  onSubmit,
  isSubmitting,
}: {
  onSubmit: (question: string) => void;
  isSubmitting: boolean;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const wasSubmitting = useRef(isSubmitting);
  // Mirrors isSubmitting but updates synchronously on click, closing the
  // window where rapid clicks land before the isSubmitting prop's
  // re-render commits and each one slips past the isSubmitting check.
  const submitInFlight = useRef(false);

  // The reader almost always has a follow-up question -- return focus to
  // the input the moment an answer finishes, without stealing it while
  // they're doing something else mid-generation.
  useEffect(() => {
    if (wasSubmitting.current && !isSubmitting) {
      textareaRef.current?.focus();
    }
    wasSubmitting.current = isSubmitting;
    submitInFlight.current = isSubmitting;
  }, [isSubmitting]);

  function submit() {
    const question = value.trim();
    if (!question || isSubmitting || submitInFlight.current) return;
    submitInFlight.current = true;
    onSubmit(question);
    setValue("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="flex items-end gap-2 border-t p-3">
      <Textarea
        ref={textareaRef}
        name="question"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about this paper..."
        aria-label="Ask a question about this paper"
        rows={2}
        className="max-h-40 min-h-16 resize-none"
      />
      <Button
        type="button"
        size="icon"
        onClick={submit}
        disabled={isSubmitting || !value.trim()}
        aria-label="Send question"
      >
        {isSubmitting ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
      </Button>
    </div>
  );
}
