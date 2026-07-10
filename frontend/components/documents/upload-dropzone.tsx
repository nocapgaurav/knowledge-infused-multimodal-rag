"use client";

import { useId, useState } from "react";
import { FileWarning, UploadCloud } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { TYPOGRAPHY } from "@/constants/typography";
import { PREPARATION_STAGE_ORDER, useUploadDocument } from "@/hooks/use-upload-document";
import { cn } from "@/lib/utils";
import { AppError } from "@/types/errors";

/**
 * The single primary action on the Landing experience (Phase 4A: "Primary
 * Action: Upload Scientific Paper"). Reports real, stage-backed progress
 * while never naming a backend stage to the user (Phase 4A: "Never
 * expose backend stages... Instead communicate: Preparing document...").
 *
 * Uses a native `<label>`/`<input type="file">` pairing rather than a
 * manual `role="button"` + keydown handler -- the browser already makes
 * this combination fully keyboard- and screen-reader-accessible, and it
 * avoids exposing two competing "button" targets for one action.
 */
export function UploadDropzone() {
  const inputId = useId();
  const [isDragging, setIsDragging] = useState(false);
  const upload = useUploadDocument();

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    upload.mutate(file);
  }

  const errorMessage =
    upload.error instanceof AppError
      ? upload.error.message
      : upload.isError
        ? "Something went wrong."
        : null;

  if (upload.isPending) {
    const stageIndex = upload.currentStage
      ? PREPARATION_STAGE_ORDER.indexOf(upload.currentStage)
      : -1;
    const progress = ((stageIndex + 1) / PREPARATION_STAGE_ORDER.length) * 100;
    return (
      <div className="flex w-full max-w-md flex-col items-center gap-3 rounded-xl border p-8 text-center">
        <UploadCloud className="text-primary size-8 animate-pulse" aria-hidden="true" />
        <p className={TYPOGRAPHY.workspaceTitle}>Preparing document...</p>
        <Progress value={progress} className="w-full" />
        <p className={TYPOGRAPHY.caption}>This usually takes under a minute.</p>
      </div>
    );
  }

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-4">
      <label
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
        className={cn(
          "has-[:focus-visible]:ring-ring/50 flex w-full cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors has-[:focus-visible]:ring-3",
          isDragging ? "border-primary bg-accent" : "border-border hover:border-primary/50",
        )}
      >
        <UploadCloud className="text-muted-foreground size-8" aria-hidden="true" />
        <div>
          <p className={TYPOGRAPHY.workspaceTitle}>Upload a scientific paper</p>
          <p className={TYPOGRAPHY.caption}>Drag and drop a PDF here, or click to browse</p>
        </div>
        <span className={buttonVariants({ variant: "default" })} aria-hidden="true">
          Choose file
        </span>
        <input
          id={inputId}
          type="file"
          accept="application/pdf"
          className="sr-only"
          onChange={(event) => handleFiles(event.target.files)}
        />
      </label>

      {errorMessage && (
        <div role="alert" className="text-error flex items-center gap-2 text-sm">
          <FileWarning className="size-4" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}
