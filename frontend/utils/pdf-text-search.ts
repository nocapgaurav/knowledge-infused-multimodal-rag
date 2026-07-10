import type { PDFDocumentProxy } from "pdfjs-dist";
import type { TextItem } from "pdfjs-dist/types/src/display/api";

/**
 * Locates the page containing a given evidence snippet by searching the
 * PDF's own real text layer -- the honest substitute for a page number
 * the backend never provides (see module12-backend-integration-gaps
 * memory). Equivalent to a reader's own Ctrl+F, not PDF parsing or
 * business logic: it never extracts structure, only matches text that
 * already exists in the client-held document.
 */
export async function findPageContainingText(
  document: PDFDocumentProxy,
  searchText: string,
): Promise<number | null> {
  const needle = normalize(searchText).slice(0, 120);
  if (!needle) return null;

  for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber++) {
    const page = await document.getPage(pageNumber);
    const content = await page.getTextContent();
    const pageText = normalize(
      content.items
        .map((item) => ("str" in (item as TextItem) ? (item as TextItem).str : ""))
        .join(" "),
    );
    if (pageText.includes(needle)) {
      return pageNumber;
    }
  }

  return null;
}

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}
