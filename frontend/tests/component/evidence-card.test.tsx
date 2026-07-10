import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";

import { EvidenceCard } from "@/components/evidence/evidence-card";
import type { EvidenceItem } from "@/types/view-models";

function buildItem(overrides: Partial<EvidenceItem> = {}): EvidenceItem {
  return {
    knowledgeUnitId: "ku-1",
    label: "KU1",
    modality: "text",
    text: "The proposed method improves accuracy by 12%.",
    groupId: "group-1",
    isPrimary: true,
    ...overrides,
  };
}

describe("EvidenceCard", () => {
  it("renders the citation label and text, never the raw knowledge unit id", () => {
    render(<EvidenceCard item={buildItem()} isActive={false} onSelect={vi.fn()} />);

    expect(screen.getByText("KU1")).toBeInTheDocument();
    expect(screen.getByText(/improves accuracy by 12%/)).toBeInTheDocument();
    expect(screen.queryByText("ku-1")).not.toBeInTheDocument();
  });

  it("calls onSelect when clicked", async () => {
    const onSelect = vi.fn();
    render(<EvidenceCard item={buildItem()} isActive={false} onSelect={onSelect} />);

    await userEvent.click(screen.getByRole("button"));

    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it("has no detectable accessibility violations", async () => {
    const { container } = render(
      <EvidenceCard item={buildItem()} isActive={false} onSelect={vi.fn()} />,
    );

    const results = await axe(container);
    expect(results.violations).toEqual([]);
  });
});
