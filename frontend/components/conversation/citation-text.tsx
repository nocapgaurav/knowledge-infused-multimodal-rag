"use client";

import { Fragment } from "react";

import { TYPOGRAPHY } from "@/constants/typography";
import { parseCitationLabels } from "@/utils/parse-citation-labels";
import type { Citation } from "@/types/view-models";

/**
 * Renders answer text with its inline citations as clickable, keyboard
 * -accessible spans (Phase 4B: "Selecting a citation should NEVER
 * navigate away from the conversation... Open Evidence, Highlight PDF").
 * Only ever shows the citation label the model used (e.g. "KU1") --
 * never a raw knowledge_unit_id.
 */
export function CitationText({
  text,
  citations,
  onSelectCitation,
}: {
  text: string;
  citations: Citation[];
  onSelectCitation: (knowledgeUnitId: string) => void;
}) {
  const tokens = parseCitationLabels(text);
  const citationsByLabel = new Map(citations.map((citation) => [citation.label, citation]));

  return (
    <p className={TYPOGRAPHY.answer}>
      {tokens.map((token, index) => {
        if (token.type === "text") return <Fragment key={index}>{token.value}</Fragment>;

        const citation = citationsByLabel.get(token.label);
        if (!citation) return <Fragment key={index}>[{token.label}]</Fragment>;

        return (
          <button
            key={index}
            type="button"
            className={TYPOGRAPHY.citation}
            onClick={() => onSelectCitation(citation.knowledgeUnitId)}
            aria-label={`View evidence ${citation.label}`}
          >
            [{citation.label}]
          </button>
        );
      })}
    </p>
  );
}
