"use client";

import { useEffect, useRef, useState } from "react";
import { Document, Page } from "react-pdf";
import { ChevronLeft, ChevronRight, FileWarning, Loader2, ZoomIn, ZoomOut } from "lucide-react";
import type { PDFDocumentProxy } from "pdfjs-dist";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

import { Button } from "@/components/ui/button";
import { TYPOGRAPHY } from "@/constants/typography";
import { loadPdfBlob } from "@/lib/pdf-storage";
import { useWorkspaceStore } from "@/store/workspace-store";
import {
  lineMatchesEvidence,
  resolveEvidenceLocation,
  stripForMatch,
} from "@/utils/evidence-locator";
import type { EvidenceTarget } from "@/types/view-models";

/**
 * The PDF viewer (Phase 4B): renders the exact bytes this browser
 * retained at upload time (see module12-backend-integration-gaps
 * memory) -- never fetched from the backend, which exposes no such
 * endpoint. Loaded only on the client via `next/dynamic({ssr:false})`
 * at its call site, since pdfjs-dist touches browser-only globals at
 * import time.
 *
 * Evidence highlighting is deterministic (Phase 4.1): the evidence's own
 * parser-recorded location (bounding boxes / page numbers) chooses the
 * page -- see `utils/evidence-locator.ts` -- and only then is precision
 * layered on top, strictly constrained to that page:
 *   1. Exact passage -- text-layer lines verbatim-contained in the
 *      evidence, marked ONLY on the resolved page (an audit showed
 *      unconstrained matching could mark the paper title while viewing
 *      a figure's evidence).
 *   2. Chunk region -- the evidence's own bounding boxes as a scaled
 *      overlay (tables/figures, whose stored text never matches a text
 *      layer).
 *   3. Page-level -- navigate with an honest note.
 *   4. Honest failure -- the "couldn't locate" message. Never a guess.
 */
export function PdfViewer({
  documentId,
  target,
}: {
  documentId: string;
  target: EvidenceTarget | null;
}) {
  const lastPage = useWorkspaceStore((state) => state.lastPdfPageByDocument[documentId]);
  const setLastPdfPage = useWorkspaceStore((state) => state.setLastPdfPage);

  const [file, setFile] = useState<Blob | null | undefined>(undefined);
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [pageNumber, setPageNumber] = useState(lastPage ?? 1);
  const [scale, setScale] = useState(1.1);
  const [resolution, setResolution] = useState<
    "idle" | "searching" | "text" | "region" | "page-only" | "not-found"
  >("idle");
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  // Persisted after commit, never inside a state updater: a Zustand write
  // notifies subscribers synchronously, and React may run updater
  // functions during render ("Cannot update a component while rendering
  // a different component" -- observed live in dev mode).
  useEffect(() => {
    setLastPdfPage(documentId, pageNumber);
  }, [documentId, pageNumber, setLastPdfPage]);

  useEffect(() => {
    let cancelled = false;
    loadPdfBlob(documentId).then((blob) => {
      if (!cancelled) setFile(blob);
    });
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const [resolvedPage, setResolvedPage] = useState<number | null>(null);

  useEffect(() => {
    if (!document || !target) return;
    let cancelled = false;
    setResolution("searching");
    resolveEvidenceLocation(document, target).then((location) => {
      if (cancelled) return;
      if (location.kind === "none") {
        setResolvedPage(null);
        setResolution("not-found");
        return;
      }
      setPageNumber(location.page);
      setResolvedPage(location.page);
      const isVisualEvidence = target.modality === "figure" || target.modality === "table";
      if (isVisualEvidence && target.boundingBoxes?.length) {
        // For a figure or table, the visual itself is the evidence -- its
        // region communicates far more than marking a caption line would.
        setResolution("region");
      } else if (location.hasTextMatch) {
        setResolution("text");
      } else if (target.boundingBoxes?.length) {
        setResolution("region");
      } else {
        setResolution("page-only");
      }
      if (process.env.NODE_ENV !== "production") {
        // Phase 4.1 debugging aid: the full citation -> highlight mapping.
        console.debug("[evidence-locator]", {
          displayLabel: target.displayLabel,
          textHead: target.text.slice(0, 60),
          pageNumbers: target.pageNumbers,
          boxPages: target.boundingBoxes?.map((box) => box.page_number),
          resolved: location,
        });
      }
    });
    return () => {
      cancelled = true;
    };
  }, [document, target]);

  // After the text layer (or region overlay) renders, bring the evidence
  // into view -- centered, without touching the user's zoom.
  const scrollHighlightIntoView = () => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const mark = container.querySelector(".pdf-evidence-mark, .pdf-evidence-region");
    mark?.scrollIntoView({ block: "center", behavior: "smooth" });
  };

  if (file === undefined) {
    return <PdfLoading />;
  }

  if (file === null) {
    return (
      <PdfMessage
        title="PDF unavailable"
        description="This document's PDF isn't available in this browser. Re-upload it to view it here."
      />
    );
  }

  const strippedEvidence = target ? stripForMatch(target.text) : null;
  // Marks may render on any page the evidence provably occupies (its own
  // box pages plus the resolved page) -- a chunk crossing a page break
  // highlights on both sides -- and never anywhere else.
  const evidencePages = new Set([
    ...(resolvedPage !== null ? [resolvedPage] : []),
    ...(target?.boundingBoxes ?? []).map((box) => box.page_number),
  ]);
  const marksActive = resolution === "text" && evidencePages.has(pageNumber);
  const regionBoxes =
    resolution === "region"
      ? (target?.boundingBoxes ?? []).filter((box) => box.page_number === pageNumber)
      : [];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b p-2">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setPageNumber((page) => Math.max(1, page - 1))}
            disabled={pageNumber <= 1}
            aria-label="Previous page"
          >
            <ChevronLeft className="size-4" />
          </Button>
          <span className={TYPOGRAPHY.caption}>
            Page {pageNumber} of {document?.numPages ?? "..."}
          </span>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setPageNumber((page) => Math.min(document?.numPages ?? page, page + 1))}
            disabled={!document || pageNumber >= document.numPages}
            aria-label="Next page"
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setScale((value) => Math.max(0.5, value - 0.15))}
            aria-label="Zoom out"
          >
            <ZoomOut className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setScale((value) => Math.min(2.5, value + 0.15))}
            aria-label="Zoom in"
          >
            <ZoomIn className="size-4" />
          </Button>
        </div>
      </div>

      {resolution === "page-only" && (
        <p className={`${TYPOGRAPHY.caption} px-3 py-1`}>
          This evidence appears on this page; its exact position could not be pinpointed.
        </p>
      )}
      {resolution === "not-found" && (
        <p className={`${TYPOGRAPHY.caption} px-3 py-1`}>
          Couldn&apos;t locate this evidence&apos;s exact text on the page -- it may span a layout
          boundary.
        </p>
      )}

      <div ref={scrollContainerRef} className="flex-1 overflow-auto">
        <Document
          file={file}
          onLoadSuccess={setDocument}
          loading={<PdfLoading />}
          error={
            <PdfMessage title="Couldn't render this PDF" description="The file may be corrupted." />
          }
          className="flex justify-center py-4"
        >
          <div className="relative">
            <Page
              pageNumber={pageNumber}
              scale={scale}
              customTextRenderer={
                marksActive && strippedEvidence
                  ? ({ str }) =>
                      lineMatchesEvidence(str, strippedEvidence)
                        ? `<mark class="pdf-evidence-mark">${escapeHtml(str)}</mark>`
                        : escapeHtml(str)
                  : undefined
              }
              onRenderTextLayerSuccess={scrollHighlightIntoView}
            />
            {regionBoxes.map((box, index) => (
              <div
                key={index}
                ref={
                  index === 0
                    ? (node) => node?.scrollIntoView({ block: "center", behavior: "smooth" })
                    : undefined
                }
                className="pdf-evidence-region pointer-events-none absolute"
                style={{
                  left: box.x0 * scale,
                  top: box.y0 * scale,
                  width: (box.x1 - box.x0) * scale,
                  height: (box.y1 - box.y0) * scale,
                }}
                aria-hidden="true"
              />
            ))}
          </div>
        </Document>
      </div>
    </div>
  );
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function PdfLoading() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2">
      <Loader2 className="text-muted-foreground size-6 animate-spin" aria-hidden="true" />
      <p className={TYPOGRAPHY.caption}>Loading PDF...</p>
    </div>
  );
}

function PdfMessage({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
      <FileWarning className="text-muted-foreground size-6" aria-hidden="true" />
      <p className={TYPOGRAPHY.body}>{title}</p>
      <p className={TYPOGRAPHY.caption}>{description}</p>
    </div>
  );
}
