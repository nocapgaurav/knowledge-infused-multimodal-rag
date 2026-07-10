/**
 * Splits answer text into plain-text and citation-label tokens (e.g.
 * "The results improved [KU1]." -> [{type:"text", ...}, {type:"citation",
 * label:"KU1"}, ...]). Pure and UI-agnostic -- rendering the tokens into
 * clickable citations is `components/conversation/citation-text.tsx`'s job.
 */

export type CitationToken = { type: "text"; value: string } | { type: "citation"; label: string };

const CITATION_PATTERN = /\[([A-Za-z0-9_-]+)\]/g;

export function parseCitationLabels(answer: string): CitationToken[] {
  const tokens: CitationToken[] = [];
  let lastIndex = 0;

  for (const match of answer.matchAll(CITATION_PATTERN)) {
    const [fullMatch, label] = match;
    const matchIndex = match.index ?? 0;
    if (matchIndex > lastIndex) {
      tokens.push({ type: "text", value: answer.slice(lastIndex, matchIndex) });
    }
    if (label) {
      tokens.push({ type: "citation", label });
    }
    lastIndex = matchIndex + fullMatch.length;
  }

  if (lastIndex < answer.length) {
    tokens.push({ type: "text", value: answer.slice(lastIndex) });
  }

  return tokens;
}
