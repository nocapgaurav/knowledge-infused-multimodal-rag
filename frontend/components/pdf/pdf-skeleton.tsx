import { TYPOGRAPHY } from "@/constants/typography";

/** A page-shaped placeholder for the moment before the PDF itself can be
 * painted (module load, byte fetch, or the first `react-pdf` render) --
 * a blank panel reads as frozen, this reads as "about to appear". */
export function PdfSkeleton() {
  return (
    <div className="flex h-full flex-col items-center gap-3 overflow-hidden p-6">
      <div className="border-border bg-card flex aspect-[8.5/11] w-full max-w-sm animate-pulse flex-col gap-2.5 rounded-md border p-6 shadow-sm">
        <div className="bg-muted h-3 w-2/3 rounded" />
        <div className="bg-muted h-2.5 w-1/3 rounded" />
        <div className="bg-muted mt-4 h-2 w-full rounded" />
        <div className="bg-muted h-2 w-11/12 rounded" />
        <div className="bg-muted h-2 w-full rounded" />
        <div className="bg-muted h-2 w-4/5 rounded" />
        <div className="bg-muted mt-4 h-2 w-full rounded" />
        <div className="bg-muted h-2 w-10/12 rounded" />
        <div className="bg-muted h-2 w-full rounded" />
      </div>
      <p className={TYPOGRAPHY.caption}>Loading PDF...</p>
    </div>
  );
}
