"use client";

import { useEffect, useRef } from "react";
import { FileText, Table2, Image as ImageIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { TYPOGRAPHY } from "@/constants/typography";
import { cn } from "@/lib/utils";
import type { EvidenceItem } from "@/types/view-models";

const MODALITY_ICON = {
  text: FileText,
  table: Table2,
  figure: ImageIcon,
} as const;

/** Renders one piece of evidence as research evidence, not raw text:
 * its structural identity (what part of the paper it is), source page,
 * honest retrieval provenance, and the original excerpt -- the excerpt
 * is never altered, only presented with context. The internal KU label
 * stays visible as a secondary badge so inline citations in the answer
 * remain traceable to their card. */
export function EvidenceCard({
  item,
  isActive,
  onSelect,
}: {
  item: EvidenceItem;
  isActive: boolean;
  onSelect: () => void;
}) {
  const Icon = MODALITY_ICON[item.modality];
  const title = item.displayLabel ?? `Passage ${item.label}`;
  const ref = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (isActive) ref.current?.scrollIntoView({ block: "nearest" });
  }, [isActive]);

  return (
    <button
      ref={ref}
      type="button"
      onClick={onSelect}
      className={cn(
        "flex w-full flex-col gap-1.5 rounded-lg border p-3 text-left transition-colors",
        isActive ? "border-evidence bg-evidence/5" : "hover:bg-muted",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <Icon className="text-evidence size-3.5 shrink-0" aria-hidden="true" />
        <span className={cn(TYPOGRAPHY.referenceLabel, "min-w-0 truncate font-medium")}>
          {title}
        </span>
        {item.pageNumbers && (
          <Badge variant="outline" className={cn(TYPOGRAPHY.caption, "shrink-0")}>
            p. {item.pageNumbers.join(", ")}
          </Badge>
        )}
        {item.displayLabel && (
          <Badge variant="outline" className={cn(TYPOGRAPHY.referenceLabel, "ml-auto shrink-0")}>
            {item.label}
          </Badge>
        )}
      </div>
      {(item.discovery ?? item.relevance) !== undefined && (
        <p className={TYPOGRAPHY.caption}>
          {item.discovery}
          {item.discovery && item.relevance !== undefined && " · "}
          {item.relevance !== undefined && `relevance ${(item.relevance * 100).toFixed(0)}%`}
        </p>
      )}
      <p className={cn(TYPOGRAPHY.evidence, "line-clamp-4")}>{item.text}</p>
    </button>
  );
}
