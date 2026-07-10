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
      { type: "citation", label: "KU1" },
      { type: "text", value: "." },
    ]);
  });

  it("extracts multiple citations in order", () => {
    expect(parseCitationLabels("See [KU1] and [KU2].")).toEqual([
      { type: "text", value: "See " },
      { type: "citation", label: "KU1" },
      { type: "text", value: " and " },
      { type: "citation", label: "KU2" },
      { type: "text", value: "." },
    ]);
  });

  it("handles a citation at the very start or end", () => {
    expect(parseCitationLabels("[KU1] starts the sentence.")).toEqual([
      { type: "citation", label: "KU1" },
      { type: "text", value: " starts the sentence." },
    ]);
    expect(parseCitationLabels("Ends with [KU1]")).toEqual([
      { type: "text", value: "Ends with " },
      { type: "citation", label: "KU1" },
    ]);
  });

  it("returns an empty array for empty input", () => {
    expect(parseCitationLabels("")).toEqual([]);
  });
});
