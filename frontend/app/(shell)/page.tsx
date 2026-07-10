import { UploadDropzone } from "@/components/documents/upload-dropzone";
import { TYPOGRAPHY } from "@/constants/typography";
import { config } from "@/lib/config";

/**
 * Landing experience (Phase 4A): the first impression, professional,
 * minimal, focused, confident -- one heading, one description, one
 * primary action, no marketing language, no feature overload.
 */
export default function LandingPage() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-8 overflow-y-auto p-8">
      <div className="max-w-md text-center">
        <h1 className={TYPOGRAPHY.appTitle}>Understand scientific papers you can trust</h1>
        <p className={`${TYPOGRAPHY.body} text-muted-foreground mt-2`}>
          Upload a paper and ask questions about it. Every answer traces back to the document
          itself.
        </p>
      </div>

      <UploadDropzone />

      <p className={TYPOGRAPHY.caption}>
        PDF only, up to {Math.round(config.upload.maxSizeBytes / (1024 * 1024))}MB. Your document is
        processed for this session and never leaves your control.
      </p>
    </div>
  );
}
