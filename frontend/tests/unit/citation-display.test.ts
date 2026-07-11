import { describe, expect, it } from "vitest";

import { compactCitationLabel } from "@/utils/citation-display";

describe("compactCitationLabel", () => {
  it("maps structural identities to compact inline labels", () => {
    expect(compactCitationLabel("Title of this paper")).toBe("Title");
    expect(compactCitationLabel("Authors and affiliations (title page)")).toBe("Authors");
    expect(compactCitationLabel("Keywords (index terms)")).toBe("Keywords");
    expect(compactCitationLabel("Abstract")).toBe("Abstract");
    expect(compactCitationLabel("Figure 2")).toBe("Figure 2");
    expect(compactCitationLabel("Table 1")).toBe("Table 1");
    expect(compactCitationLabel("Bibliography reference [14]")).toBe("Ref 14");
  });

  it("compacts numbered section titles to their numbering", () => {
    expect(compactCitationLabel("Section: III. PROPOSED METHODOLOGY")).toBe("§ III");
    expect(compactCitationLabel("Section: 3.2 Encoding")).toBe("§ 3.2");
  });

  it("truncates long unnumbered section titles", () => {
    expect(compactCitationLabel("Section: Comparative Analysis Of Everything Ever")).toBe(
      "Comparative Analysis O…",
    );
  });
});
