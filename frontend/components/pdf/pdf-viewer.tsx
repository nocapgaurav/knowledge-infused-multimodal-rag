"use client";

import { useEffect, useState } from "react";
import { Document, Page } from "react-pdf";
import { ChevronLeft, ChevronRight, FileWarning, Loader2, ZoomIn, ZoomOut } from "lucide-react";
import type { PDFDocumentProxy } from "pdfjs-dist";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

import { Button } from "@/components/ui/button";
import { TYPOGRAPHY } from "@/constants/typography";
import { loadPdfBlob } from "@/lib/pdf-storage";
import { useWorkspaceStore } from "@/store/workspace-store";
import { findPageContainingText } from "@/utils/pdf-text-search";

/**
 * The PDF viewer (Phase 4B): renders the exact bytes this browser
 * retained at upload time (see module12-backend-integration-gaps
 * memory) -- never fetched from the backend, which exposes no such
 * endpoint. Loaded only on the client via `next/dynamic({ssr:false})`
 * at its call site, since pdfjs-dist touches browser-only globals at
 * import time.
 */
export function PdfViewer({
  documentId,
  searchText,
}: {
  documentId: string;
  searchText: string | null;
}) {
  const lastPage = useWorkspaceStore((state) => state.lastPdfPageByDocument[documentId]);
  const setLastPdfPage = useWorkspaceStore((state) => state.setLastPdfPage);

  const [file, setFile] = useState<Blob | null | undefined>(undefined);
  const [document, setDocument] = useState<PDFDocumentProxy | null>(null);
  const [pageNumber, setPageNumber] = useState(lastPage ?? 1);
  const [scale, setScale] = useState(1.1);
  const [searchStatus, setSearchStatus] = useState<"idle" | "searching" | "not-found">("idle");

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

  useEffect(() => {
    if (!document || !searchText) return;
    let cancelled = false;
    setSearchStatus("searching");
    findPageContainingText(document, searchText).then((page) => {
      if (cancelled) return;
      if (page !== null) {
        setPageNumber(page);
        setSearchStatus("idle");
      } else {
        setSearchStatus("not-found");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [document, searchText, setPageNumber]);

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

      {searchStatus === "not-found" && (
        <p className={`${TYPOGRAPHY.caption} px-3 py-1`}>
          Couldn&apos;t locate this evidence&apos;s exact text on the page -- it may span a layout
          boundary.
        </p>
      )}

      <div className="flex-1 overflow-auto">
        <Document
          file={file}
          onLoadSuccess={setDocument}
          loading={<PdfLoading />}
          error={
            <PdfMessage title="Couldn't render this PDF" description="The file may be corrupted." />
          }
          className="flex justify-center py-4"
        >
          <Page pageNumber={pageNumber} scale={scale} />
        </Document>
      </div>
    </div>
  );
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
