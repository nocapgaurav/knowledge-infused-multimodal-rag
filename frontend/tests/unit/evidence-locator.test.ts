import { describe, expect, it } from "vitest";
import type { PDFDocumentProxy } from "pdfjs-dist";

import {
  lineMatchesEvidence,
  resolveEvidenceLocation,
  stripForMatch,
} from "@/utils/evidence-locator";

function fakeDocument(pages: string[][]): PDFDocumentProxy {
  return {
    numPages: pages.length,
    getPage: (n: number) =>
      Promise.resolve({
        getTextContent: () =>
          Promise.resolve({ items: (pages[n - 1] ?? []).map((str) => ({ str })) }),
      }),
  } as unknown as PDFDocumentProxy;
}

describe("lineMatchesEvidence", () => {
  const evidence = stripForMatch(
    "Current approaches have primarily focused on one modality at a time and/or " +
      "treated each modality as separate and independent.",
  );

  it("matches hyphen-broken text-layer lines", () => {
    expect(
      lineMatchesEvidence("at a time and/or treated each modality as separate and indepen-", evidence),
    ).toBe(true);
  });

  it("never matches unrelated text", () => {
    expect(lineMatchesEvidence("Scientific Document Question Answering", evidence)).toBe(false);
  });

  it("ignores short common fragments", () => {
    expect(lineMatchesEvidence("modality", evidence)).toBe(false);
  });
});

describe("resolveEvidenceLocation", () => {
  const doc = fakeDocument([
    ["Knowledge-Infused Multimodal Question", "Answering System"],
    ["An open system that integrates different kinds of data from", "research papers is proposed."],
    ["Further, there is a need for improving how well the QA", "system performs under adverse conditions."],
  ]);

  it("prefers the bounding-box page whose text contains the evidence", async () => {
    // A split chunk carrying its sibling's boxes too: page 2 listed first,
    // but the chunk's own text lives on page 3.
    const location = await resolveEvidenceLocation(doc, {
      text: "Further, there is a need for improving how well the QA system performs under adverse conditions.",
      boundingBoxes: [
        { page_number: 2, x0: 0, y0: 0, x1: 1, y1: 1 },
        { page_number: 3, x0: 0, y0: 0, x1: 1, y1: 1 },
      ],
    });
    expect(location).toEqual({
      kind: "located",
      page: 3,
      hasTextMatch: true,
      source: "bounding-boxes",
    });
  });

  it("falls back to the first box page without text match (tables/figures)", async () => {
    const location = await resolveEvidenceLocation(doc, {
      text: "| Author | Year | markdown that never matches a text layer |",
      boundingBoxes: [{ page_number: 2, x0: 0, y0: 0, x1: 1, y1: 1 }],
    });
    expect(location).toEqual({
      kind: "located",
      page: 2,
      hasTextMatch: false,
      source: "bounding-boxes",
    });
  });

  it("only searches the whole document when no location metadata exists", async () => {
    const location = await resolveEvidenceLocation(doc, {
      text: "An open system that integrates different kinds of data from research papers is proposed.",
    });
    expect(location).toEqual({
      kind: "located",
      page: 2,
      hasTextMatch: true,
      source: "text-search",
    });
  });

  it("reports none honestly when nothing locates the evidence", async () => {
    const location = await resolveEvidenceLocation(doc, {
      text: "Text that exists nowhere in this document at all, not even close.",
    });
    expect(location).toEqual({ kind: "none" });
  });
});
