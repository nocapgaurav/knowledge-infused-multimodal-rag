"use client";

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

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "flex w-full flex-col gap-1.5 rounded-lg border p-3 text-left transition-colors",
        isActive ? "border-evidence bg-evidence/5" : "hover:bg-muted",
      )}
    >
      <div className="flex items-center gap-2">
        <Icon className="text-evidence size-3.5 shrink-0" aria-hidden="true" />
        <Badge variant="outline" className={TYPOGRAPHY.referenceLabel}>
          {item.label}
        </Badge>
        <span className={cn(TYPOGRAPHY.caption, "capitalize")}>{item.modality}</span>
      </div>
      <p className={cn(TYPOGRAPHY.evidence, "line-clamp-4")}>{item.text}</p>
    </button>
  );
}
