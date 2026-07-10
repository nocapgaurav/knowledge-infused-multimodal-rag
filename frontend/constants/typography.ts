/**
 * The application's single typography scale (Phase 3: "Never choose
 * arbitrary font sizes. Maintain one typography scale throughout the
 * application."). Every text-bearing component composes its className
 * from one of these named levels instead of inventing size/weight/leading
 * combinations at the call site.
 */
export const TYPOGRAPHY = {
  appTitle: "text-lg font-semibold tracking-tight",
  workspaceTitle: "text-base font-semibold tracking-tight",
  panelTitle: "text-sm font-semibold tracking-tight",
  sectionTitle: "text-sm font-medium text-foreground/90",
  answer: "text-[0.95rem] leading-7 text-foreground",
  body: "text-sm leading-6 text-foreground",
  metadata: "text-xs text-metadata",
  caption: "text-xs text-muted-foreground",
  evidence: "text-sm leading-6 text-foreground",
  referenceLabel: "text-xs font-medium text-reference",
  citation:
    "text-xs font-medium text-evidence underline decoration-evidence/40 decoration-2 underline-offset-2",
  code: "font-mono text-[0.85em]",
} as const;

export type TypographyLevel = keyof typeof TYPOGRAPHY;
