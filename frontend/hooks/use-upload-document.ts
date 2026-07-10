"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";

import { config } from "@/lib/config";
import { savePdfBlob } from "@/lib/pdf-storage";
import {
  prepareDocument,
  uploadDocument,
  type PreparationStage,
} from "@/services/document-service";
import { useDocumentLibraryStore } from "@/store/document-library-store";
import { useWorkspaceStore } from "@/store/workspace-store";
import { AppError } from "@/types/errors";

export const PREPARATION_STAGE_ORDER: PreparationStage[] = [
  "parsing",
  "representing",
  "embedding",
  "indexing",
  "graphing",
];

const LARGE_FILE_WARNING_THRESHOLD_BYTES = 20 * 1024 * 1024;

/**
 * The full "upload -> prepare -> ready" flow (Phase 4A), presented to
 * callers as one operation. Internally this is: upload, retain the exact
 * bytes locally for the PDF viewer to render later, then sequence the
 * five real pipeline calls -- never a single backend call, because no
 * such call exists (see module12-backend-integration-gaps memory).
 */
export function useUploadDocument() {
  const router = useRouter();
  const [currentStage, setCurrentStage] = useState<PreparationStage | null>(null);
  const addDocument = useDocumentLibraryStore((state) => state.addDocument);
  const setPipelineStage = useDocumentLibraryStore((state) => state.setPipelineStage);
  const setStatus = useDocumentLibraryStore((state) => state.setStatus);
  const selectDocument = useWorkspaceStore((state) => state.selectDocument);

  const mutation = useMutation({
    mutationFn: async (file: File) => {
      setCurrentStage(null);
      if (!navigator.onLine) {
        throw new AppError("You're offline. Reconnect to upload a document.", "network");
      }
      if (!config.upload.acceptedMimeTypes.includes(file.type)) {
        throw new AppError("Only PDF files are supported.", "validation");
      }
      if (file.size > config.upload.maxSizeBytes) {
        throw new AppError(
          `This file is larger than the ${Math.round(config.upload.maxSizeBytes / (1024 * 1024))}MB limit.`,
          "validation",
        );
      }
      if (file.size > LARGE_FILE_WARNING_THRESHOLD_BYTES) {
        toast.warning("This is a large document", {
          description: "Preparing it may take longer than usual.",
        });
      }

      const { document_id: documentId } = await uploadDocument(file);
      addDocument(documentId, file.name);
      toast.success("Document uploaded");
      await savePdfBlob(documentId, file);

      try {
        await prepareDocument(documentId, (stage) => {
          setPipelineStage(documentId, stage);
          setCurrentStage(stage);
        });
        setStatus(documentId, "ready");
        toast.success("Document ready");
      } catch (error) {
        const message =
          error instanceof AppError ? error.message : "Preparing this document failed.";
        setStatus(documentId, "failed", message);
        toast.error("Couldn't prepare this document", { description: message });
        throw error;
      }

      return documentId;
    },
    onSuccess: (documentId) => {
      selectDocument(documentId);
      router.push(`/workspace/${documentId}`);
    },
  });

  return { ...mutation, currentStage };
}
