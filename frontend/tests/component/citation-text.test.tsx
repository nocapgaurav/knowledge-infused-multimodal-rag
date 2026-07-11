import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CitationText } from "@/components/conversation/citation-text";
import type { Citation } from "@/types/view-models";

describe("CitationText", () => {
  const citations: Citation[] = [{ label: "KU1", knowledgeUnitId: "ku-1", textExcerpt: "excerpt" }];

  it("renders a clickable citation for a recognized label", async () => {
    const onSelectCitation = vi.fn();
    render(
      <CitationText
        text="The results improved [KU1]."
        citations={citations}
        onSelectCitation={onSelectCitation}
      />,
    );

    const citationButton = screen.getByRole("button", { name: "View evidence KU1" });
    await userEvent.click(citationButton);

    expect(onSelectCitation).toHaveBeenCalledWith(
      expect.objectContaining({ knowledgeUnitId: "ku-1" }),
    );
  });

  it("renders an unresolved citation label as plain text, not a button", () => {
    const { container } = render(
      <CitationText text="See [KU9]." citations={citations} onSelectCitation={vi.fn()} />,
    );

    expect(container.textContent).toBe("See [KU9].");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("never exposes the raw knowledge_unit_id in the rendered text", () => {
    render(
      <CitationText
        text="The results improved [KU1]."
        citations={citations}
        onSelectCitation={vi.fn()}
      />,
    );

    expect(screen.queryByText(/ku-1/)).not.toBeInTheDocument();
  });
});
