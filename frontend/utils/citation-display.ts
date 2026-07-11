/**
 * Compact, human-readable inline labels for citation buttons.
 *
 * The backend's `display_label` is a full structural identity ("Authors
 * and affiliations (title page)", "Section: III. PROPOSED METHODOLOGY").
 * Inline citations need something a reader can scan mid-sentence, so this
 * maps each identity family to its shortest honest form. Falls back to
 * the model's own label (KUn) when no identity is known -- the KU label
 * also stays in the accessible name for traceability either way.
 */

const SECTION_PREFIX = "Section: ";
const SECTION_NUMBERING = /^([IVXLCDM]+|\d+(?:\.\d+)*)\.?\s/;
const MAX_SECTION_TITLE_LENGTH = 22;

export function compactCitationLabel(displayLabel: string): string {
  if (displayLabel.startsWith("Title of this paper")) return "Title";
  if (displayLabel.startsWith("Authors and affiliations")) return "Authors";
  if (displayLabel.startsWith("Keywords")) return "Keywords";
  const reference = displayLabel.match(/^Bibliography reference \[(\d+)\]/);
  if (reference) return `Ref ${reference[1]}`;
  if (displayLabel.startsWith(SECTION_PREFIX)) {
    const title = displayLabel.slice(SECTION_PREFIX.length).trim();
    const numbering = title.match(SECTION_NUMBERING);
    if (numbering) return `§ ${numbering[1]}`;
    return title.length > MAX_SECTION_TITLE_LENGTH
      ? `${title.slice(0, MAX_SECTION_TITLE_LENGTH).trimEnd()}…`
      : title;
  }
  return displayLabel;
}
