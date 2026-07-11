import { describe, expect, it } from "vitest";

import { parseCitationLabels } from "@/utils/parse-citation-labels";

describe("parseCitationLabels", () => {
  it("splits plain text with no citations into a single text token", () => {
    expect(parseCitationLabels("No citations here.")).toEqual([
      { type: "text", value: "No citations here." },
    ]);
  });

  it("extracts a single inline citation", () => {
    expect(parseCitationLabels("The results improved [KU1].")).toEqual([
      { type: "text", value: "The results improved " },
      { type: "citation", label: "KU1", raw: "[KU1]" },
      { type: "text", value: "." },
    ]);
  });

  it("extracts multiple citations in order", () => {
    expect(parseCitationLabels("See [KU1] and [KU2].")).toEqual([
      { type: "text", value: "See " },
      { type: "citation", label: "KU1", raw: "[KU1]" },
      { type: "text", value: " and " },
      { type: "citation", label: "KU2", raw: "[KU2]" },
      { type: "text", value: "." },
    ]);
  });

  it("handles a citation at the very start or end", () => {
    expect(parseCitationLabels("[KU1] starts the sentence.")).toEqual([
      { type: "citation", label: "KU1", raw: "[KU1]" },
      { type: "text", value: " starts the sentence." },
    ]);
    expect(parseCitationLabels("Ends with [KU1]")).toEqual([
      { type: "text", value: "Ends with " },
      { type: "citation", label: "KU1", raw: "[KU1]" },
    ]);
  });

  it("extracts parenthesized KU citations, as real models emit them", () => {
    expect(parseCitationLabels("The system combines modalities (KU4).")).toEqual([
      { type: "text", value: "The system combines modalities " },
      { type: "text", value: "(" },
      { type: "citation", label: "KU4", raw: "KU4" },
      { type: "text", value: ")" },
      { type: "text", value: "." },
    ]);
  });

  it("extracts every label from a comma-separated citation list", () => {
    expect(parseCitationLabels("combines text and tables (KU4, KU8).")).toEqual([
      { type: "text", value: "combines text and tables " },
      { type: "text", value: "(" },
      { type: "citation", label: "KU4", raw: "KU4" },
      { type: "text", value: ", " },
      { type: "citation", label: "KU8", raw: "KU8" },
      { type: "text", value: ")" },
      { type: "text", value: "." },
    ]);
  });

  it("extracts every label from a bracketed citation list", () => {
    expect(parseCitationLabels("shown earlier [KU1, KU2]")).toEqual([
      { type: "text", value: "shown earlier " },
      { type: "text", value: "[" },
      { type: "citation", label: "KU1", raw: "KU1" },
      { type: "text", value: ", " },
      { type: "citation", label: "KU2", raw: "KU2" },
      { type: "text", value: "]" },
    ]);
  });

  it("does not treat ordinary prose parentheses as citations", () => {
    expect(parseCitationLabels("As shown before (Doe, 2021) and (see Table 1).")).toEqual([
      { type: "text", value: "As shown before (Doe, 2021) and (see Table 1)." },
    ]);
  });

  it("handles mixed bracket and parenthesis citations in one answer", () => {
    expect(parseCitationLabels("First [KU1], then (KU2).")).toEqual([
      { type: "text", value: "First " },
      { type: "citation", label: "KU1", raw: "[KU1]" },
      { type: "text", value: ", then " },
      { type: "text", value: "(" },
      { type: "citation", label: "KU2", raw: "KU2" },
      { type: "text", value: ")" },
      { type: "text", value: "." },
    ]);
  });

  it("returns an empty array for empty input", () => {
    expect(parseCitationLabels("")).toEqual([]);
  });
});
