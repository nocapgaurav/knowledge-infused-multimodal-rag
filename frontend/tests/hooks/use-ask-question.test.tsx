import { act, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useAskQuestion } from "@/hooks/use-ask-question";
import * as generationService from "@/services/generation-service";
import { useConversationStore } from "@/store/conversation-store";
import { AppError } from "@/types/errors";
import { renderWithProviders } from "../test-utils";

const DOCUMENT_ID = "doc-1";

function Harness({ documentId }: { documentId: string }) {
  const ask = useAskQuestion(documentId);
  return <button onClick={() => ask.mutate("What are the main results?")}>ask</button>;
}

describe("useAskQuestion", () => {
  beforeEach(() => {
    useConversationStore.setState({ turnsByDocument: {} });
    vi.restoreAllMocks();
    Object.defineProperty(window.navigator, "onLine", { value: true, configurable: true });
  });

  it("records a pending turn immediately, then completes it on success", async () => {
    vi.spyOn(generationService, "generateAnswer").mockResolvedValue({
      response: {} as never,
      answer: {
        answer: "Accuracy improved [KU1].",
        executiveSummary: "Accuracy improved.",
        confidence: 0.9,
        answerStatus: "sufficient_evidence",
        citations: [],
        evidenceItems: [],
        limitations: [],
        warnings: [],
        references: [],
      },
    });

    const { getByRole } = renderWithProviders(<Harness documentId={DOCUMENT_ID} />);

    await act(async () => {
      getByRole("button").click();
    });

    await waitFor(() => {
      const turns = useConversationStore.getState().turnsByDocument[DOCUMENT_ID] ?? [];
      expect(turns).toHaveLength(1);
      expect(turns[0]?.status).toBe("complete");
      expect(turns[0]?.answer?.answer).toContain("Accuracy improved");
    });
  });

  it("records a failed turn without throwing out of the mutation when offline", async () => {
    Object.defineProperty(window.navigator, "onLine", { value: false, configurable: true });

    const { getByRole } = renderWithProviders(<Harness documentId={DOCUMENT_ID} />);

    await act(async () => {
      getByRole("button").click();
    });

    await waitFor(() => {
      const turns = useConversationStore.getState().turnsByDocument[DOCUMENT_ID] ?? [];
      expect(turns[0]?.status).toBe("failed");
      expect(turns[0]?.failureReason).toMatch(/offline/i);
    });
  });

  it("surfaces the backend's AppError message on failure", async () => {
    vi.spyOn(generationService, "generateAnswer").mockRejectedValue(
      new AppError("Grounding failed for this question.", "server"),
    );

    const { getByRole } = renderWithProviders(<Harness documentId={DOCUMENT_ID} />);

    await act(async () => {
      getByRole("button").click();
    });

    await waitFor(() => {
      const turns = useConversationStore.getState().turnsByDocument[DOCUMENT_ID] ?? [];
      expect(turns[0]?.failureReason).toBe("Grounding failed for this question.");
    });
  });
});
