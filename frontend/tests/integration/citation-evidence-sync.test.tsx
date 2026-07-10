import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { ConversationPanel } from "@/components/conversation/conversation-panel";
import { EvidenceList } from "@/components/evidence/evidence-list";
import { useConversationStore } from "@/store/conversation-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import { renderWithProviders } from "../test-utils";
import type { ConversationTurn } from "@/types/view-models";

const DOCUMENT_ID = "doc-1";

const TURN: ConversationTurn = {
  id: "turn-1",
  question: "What are the main results?",
  askedAt: "2026-01-01T00:00:00Z",
  status: "complete",
  answer: {
    answer: "Accuracy improved substantially [KU1].",
    executiveSummary: "Accuracy improved.",
    confidence: 0.9,
    answerStatus: "sufficient_evidence",
    citations: [{ label: "KU1", knowledgeUnitId: "ku-1", textExcerpt: "excerpt" }],
    evidenceItems: [
      {
        knowledgeUnitId: "ku-1",
        label: "KU1",
        modality: "text",
        text: "Accuracy improved from 71% to 89%.",
        groupId: "ku-1",
        isPrimary: true,
      },
    ],
    limitations: [],
    warnings: [],
    references: [],
  },
};

/**
 * Exercises the real primary workflow loop (Phase 4B/4D): clicking a
 * citation in the conversation must update the shared workspace store,
 * and the independent Evidence panel -- reading that same real store,
 * never a mock of it -- must reflect the selection immediately.
 */
describe("citation click -> evidence panel sync", () => {
  beforeEach(() => {
    useConversationStore.setState({ turnsByDocument: { [DOCUMENT_ID]: [TURN] } });
    useWorkspaceStore.setState({ openedEvidenceId: null });
  });

  it("highlights the corresponding evidence card when its citation is clicked", async () => {
    renderWithProviders(
      <>
        <ConversationPanel documentId={DOCUMENT_ID} />
        <EvidenceList documentId={DOCUMENT_ID} />
      </>,
    );

    expect(useWorkspaceStore.getState().openedEvidenceId).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "View evidence KU1" }));

    expect(useWorkspaceStore.getState().openedEvidenceId).toBe("ku-1");
    const evidenceCard = screen.getByText(/Accuracy improved from 71%/).closest("button");
    expect(evidenceCard).toHaveClass("border-evidence");
  });
});
