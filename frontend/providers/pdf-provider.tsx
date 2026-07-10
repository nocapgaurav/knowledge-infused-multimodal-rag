"use client";

import { useEffect, type ReactNode } from "react";

/**
 * Configures PDF.js's worker once for the whole app. Every `<Document>`
 * from react-pdf elsewhere in the app relies on this having already run.
 *
 * `react-pdf`/`pdfjs-dist` reference browser-only globals (e.g.
 * `DOMMatrix`) at module-evaluation time, which breaks Next's server-side
 * render pass even inside a "use client" component -- so the import
 * itself, not just its use, must be deferred to a browser-only effect.
 */
export function PdfProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    void import("react-pdf").then(({ pdfjs }) => {
      pdfjs.GlobalWorkerOptions.workerSrc = new URL(
        "pdfjs-dist/build/pdf.worker.min.mjs",
        import.meta.url,
      ).toString();
    });
  }, []);

  return children;
}
