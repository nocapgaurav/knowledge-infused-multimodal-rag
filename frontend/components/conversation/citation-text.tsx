"use client";

import { Fragment } from "react";

import { TYPOGRAPHY } from "@/constants/typography";
import { compactCitationLabel } from "@/utils/citation-display";
import { parseCitationLabels } from "@/utils/parse-citation-labels";
import type { Citation } from "@/types/view-models";

/**
 * Renders answer text with its inline citations as clickable, keyboard
 * -accessible spans (Phase 4B: "Selecting a citation should NEVER
 * navigate away from the conversation... Open Evidence, Highlight PDF").
 * The visible label is the evidence's human identity ("Figure 2",
 * "Authors") when the backend knows it -- internal KU labels remain in
 * the accessible name for traceability, and are the fallback when no
 * identity is known. Never a raw knowledge_unit_id.
 */
export function CitationText({
  text,
  citations,
  onSelectCitation,
}: {
  text: string;
  citations: Citation[];
  onSelectCitation: (citation: Citation) => void;
}) {
  const tokens = parseCitationLabels(text);
  const citationsByLabel = new Map(citations.map((citation) => [citation.label, citation]));

  return (
    <p className={TYPOGRAPHY.answer}>
      {tokens.map((token, index) => {
        if (token.type === "text") return <Fragment key={index}>{token.value}</Fragment>;

        const citation = citationsByLabel.get(token.label);
        if (!citation) return <Fragment key={index}>{token.raw}</Fragment>;

        const visible = citation.displayLabel
          ? `[${compactCitationLabel(citation.displayLabel)}]`
          : token.raw;
        return (
          <button
            key={index}
            type="button"
            className={TYPOGRAPHY.citation}
            onClick={() => onSelectCitation(citation)}
            aria-label={`View evidence ${citation.label}${
              citation.displayLabel ? ` (${citation.displayLabel})` : ""
            }`}
          >
            {visible}
          </button>
        );
      })}
    </p>
  );
}
