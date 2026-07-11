/**
 * Splits answer text into plain-text and citation-label tokens (e.g.
 * "The results improved [KU1]." -> [{type:"text", ...}, {type:"citation",
 * label:"KU1"}, ...]). Pure and UI-agnostic -- rendering the tokens into
 * clickable citations is `components/conversation/citation-text.tsx`'s job.
 *
 * The citation shapes recognized mirror the backend's own citation
 * resolver: `[LABEL]` (the format the model is asked for), plus the
 * delimiter substitutions real models make often enough to matter --
 * observed live with qwen2.5: `(KU1)` and comma-separated lists like
 * `(KU4, KU8)` or `[KU4, KU8]`. Delimited groups are restricted to
 * KU-prefixed labels so ordinary prose parentheses like "(Doe, 2021)"
 * are never treated as citations. Each citation token carries the exact
 * matched text (`raw`) so renderers reproduce the answer verbatim; for a
 * list, the surrounding delimiters and separators become text tokens and
 * each label becomes its own citation token.
 */

export type CitationToken =
  { type: "text"; value: string } | { type: "citation"; label: string; raw: string };

const CITATION_PATTERN = /\[([A-Za-z0-9_-]+)\]|[[(]\s*(KU\d+(?:\s*,\s*KU\d+)*)\s*[\])]/g;

const LABEL_PATTERN = /KU\d+/g;

export function parseCitationLabels(answer: string): CitationToken[] {
  const tokens: CitationToken[] = [];
  let lastIndex = 0;

  for (const match of answer.matchAll(CITATION_PATTERN)) {
    const [fullMatch, bracketedLabel, labelGroup] = match;
    const matchIndex = match.index ?? 0;
    if (matchIndex > lastIndex) {
      tokens.push({ type: "text", value: answer.slice(lastIndex, matchIndex) });
    }

    if (bracketedLabel) {
      tokens.push({ type: "citation", label: bracketedLabel, raw: fullMatch });
    } else if (labelGroup) {
      pushGroupTokens(tokens, fullMatch);
    }

    lastIndex = matchIndex + fullMatch.length;
  }

  if (lastIndex < answer.length) {
    tokens.push({ type: "text", value: answer.slice(lastIndex) });
  }

  return tokens;
}

/** Tokenizes a delimited label group like "(KU4, KU8)" into text tokens
 * for the delimiters/separators and one citation token per label, so the
 * rendered output reads exactly like the original answer text. */
function pushGroupTokens(tokens: CitationToken[], group: string): void {
  let cursor = 0;
  for (const labelMatch of group.matchAll(LABEL_PATTERN)) {
    const labelIndex = labelMatch.index ?? 0;
    if (labelIndex > cursor) {
      tokens.push({ type: "text", value: group.slice(cursor, labelIndex) });
    }
    tokens.push({ type: "citation", label: labelMatch[0], raw: labelMatch[0] });
    cursor = labelIndex + labelMatch[0].length;
  }
  if (cursor < group.length) {
    tokens.push({ type: "text", value: group.slice(cursor) });
  }
}
