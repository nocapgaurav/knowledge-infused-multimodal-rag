import type { PDFDocumentProxy } from "pdfjs-dist";
import type { TextItem } from "pdfjs-dist/types/src/display/api";

import type { EvidenceTarget } from "@/types/view-models";

/**
 * Deterministic evidence-to-PDF location resolution (Phase 4.1).
 *
 * The canonical location source is the evidence's own parser-recorded
 * metadata (bounding boxes and page numbers), which flows unchanged from
 * chunking through retrieval to the clicked citation. The viewer never
 * guesses globally: a full-document text search happens ONLY when the
 * evidence carries no location metadata at all (last resort), exactly as
 * the precision ladder requires.
 *
 * Text matching is hyphenation-tolerant: PDF text layers break words at
 * line ends ("indepen-" / "dent"), which an exhaustive audit against a
 * real paper showed defeated exact matching for 31% of chunks, silently
 * demoting them to coarser fallbacks. Comparing both sides with hyphens
 * removed is a deterministic, symmetric transform -- it can never match
 * text the evidence does not contain.
 */

export type EvidenceLocation =
  | {
      kind: "located";
      page: number;
      /** Whether the page's text layer verbatim-contains the evidence
       * opening -- when false, the region/page fallbacks render instead
       * of text marks. */
      hasTextMatch: boolean;
      /** How the page was chosen -- for debugging and honest fallback
       * messaging. */
      source: "bounding-boxes" | "page-numbers" | "text-search";
    }
  | { kind: "none" };

const NEEDLE_LENGTH = 120;
const MINIMUM_MATCH_LENGTH = 18;

export function normalizeForMatch(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

/** Normalized text made insensitive to PDF typography, symmetrically on
 * both sides: line-break hyphenation is collapsed ("sys- tem" ->
 * "system"), remaining hyphens/dashes removed ("cross-modal" ==
 * "cross- modal"), and quote glyphs dropped (chunks store straight
 * quotes, PDFs render curly ones). Pure character deletions applied to
 * both haystack and needle can never match text the evidence does not
 * contain. */
export function stripForMatch(text: string): string {
  return normalizeForMatch(text)
    .replace(/[-–—]\s+/g, "")
    .replace(/[-–—]/g, "")
    .replace(/["'\u201c\u201d\u2018\u2019]/g, "");
}

/** Whether one text-layer line belongs to the evidence passage.
 * A line is marked only when its stripped text (at least 18 characters)
 * appears verbatim inside the stripped evidence -- accuracy over
 * precision: text merely resembling the evidence never matches. */
export function lineMatchesEvidence(line: string, strippedEvidence: string): boolean {
  const stripped = stripForMatch(line);
  if (stripped.length < MINIMUM_MATCH_LENGTH) return false;
  return strippedEvidence.includes(stripped);
}

/** Resolve where in the PDF a piece of evidence lives.
 *
 * Order: (1) the evidence's own bounding-box pages, picking the page
 * whose text actually contains the evidence opening (split chunks carry
 * their whole block's boxes, so the first box may belong to a sibling
 * part); (2) recorded page numbers, same test; (3) full-document text
 * search, last resort, only when no location metadata exists; (4) an
 * honest "none".
 */
export async function resolveEvidenceLocation(
  document: PDFDocumentProxy,
  target: EvidenceTarget,
): Promise<EvidenceLocation> {
  const strippedEvidence = stripForMatch(target.text);
  const needle = strippedEvidence.slice(0, NEEDLE_LENGTH);

  const boxPages = uniquePages((target.boundingBoxes ?? []).map((box) => box.page_number));
  const recordedPages = uniquePages(target.pageNumbers ?? []);

  for (const [pages, source] of [
    [boxPages, "bounding-boxes"],
    [recordedPages, "page-numbers"],
  ] as const) {
    if (!pages.length) continue;
    if (needle) {
      for (const page of pages) {
        if ((await pageStrippedText(document, page)).includes(needle)) {
          return { kind: "located", page, hasTextMatch: true, source };
        }
      }
    }
    const firstPage = pages[0];
    if (firstPage !== undefined) {
      return { kind: "located", page: firstPage, hasTextMatch: false, source };
    }
  }

  if (needle) {
    for (let page = 1; page <= document.numPages; page++) {
      if ((await pageStrippedText(document, page)).includes(needle)) {
        return { kind: "located", page, hasTextMatch: true, source: "text-search" };
      }
    }
  }
  return { kind: "none" };
}

function uniquePages(pages: number[]): number[] {
  return [...new Set(pages)].filter((page) => Number.isInteger(page) && page > 0);
}

async function pageStrippedText(document: PDFDocumentProxy, page: number): Promise<string> {
  if (page < 1 || page > document.numPages) return "";
  const pdfPage = await document.getPage(page);
  const content = await pdfPage.getTextContent();
  return stripForMatch(
    content.items
      .map((item) => ("str" in (item as TextItem) ? (item as TextItem).str : ""))
      .join(" "),
  );
}
