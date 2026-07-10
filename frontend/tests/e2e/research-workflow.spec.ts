import path from "path";
import { expect, test } from "@playwright/test";

/**
 * The real, no-mock, full-stack end-to-end test (Phase 5: "These tests
 * should execute against the real backend whenever possible"). Drives
 * the actual primary workflow (Phase 4D) against the live Modules 1-11
 * backend, real Qdrant, real Neo4j, and a real local Ollama model --
 * nothing in this file is faked.
 */

const SAMPLE_PDF = path.resolve(__dirname, "../../../tests/parser/fixtures/sample_paper.pdf");

test.describe("primary research workflow against the real backend", () => {
  test("upload -> prepare -> ask -> evidence -> PDF sync", async ({ page }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Understand scientific papers you can trust" }),
    ).toBeVisible();

    // Upload the real fixture PDF.
    await page.locator('input[type="file"]').setInputFiles(SAMPLE_PDF);

    // Real preparation: parse -> represent -> embed -> index -> graph.
    await expect(page.getByText("Preparing document...")).toBeVisible();
    await expect(page).toHaveURL(/\/workspace\//, { timeout: 120_000 });

    // Ask a real question, answered by the real generation pipeline.
    const questionBox = page.getByRole("textbox", { name: "Ask a question about this paper" });
    await questionBox.fill("What are the main results described in this paper?");
    await questionBox.press("Enter");

    await expect(page.getByText("Thinking through the evidence...")).toBeVisible();
    // Generation runs a real local LLM call -- allow ample real time for it.
    await expect(page.getByText("Thinking through the evidence...")).toBeHidden({
      timeout: 120_000,
    });

    // A real grounded answer with at least one real citation should appear.
    const citation = page.getByRole("button", { name: /View evidence/ }).first();
    await expect(citation).toBeVisible();

    // Clicking a citation opens real evidence and switches to the PDF tab.
    await citation.click();
    await expect(page.getByRole("tab", { name: "PDF", selected: true })).toBeVisible();

    // The PDF viewer renders the browser's own retained copy of the real file.
    await expect(page.getByText(/Page \d+ of \d+/)).toBeVisible({ timeout: 15_000 });
  });
});
