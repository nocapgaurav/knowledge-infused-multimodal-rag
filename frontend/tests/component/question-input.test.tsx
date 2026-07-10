import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";

import { QuestionInput } from "@/components/conversation/question-input";

describe("QuestionInput", () => {
  it("submits the trimmed question on Enter and clears the input", async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} isSubmitting={false} />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "  What are the main results?  ");
    await userEvent.keyboard("{Enter}");

    expect(onSubmit).toHaveBeenCalledWith("What are the main results?");
    expect(textarea).toHaveValue("");
  });

  it("does not submit on Shift+Enter, allowing a newline instead", async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} isSubmitting={false} />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "line one");
    await userEvent.keyboard("{Shift>}{Enter}{/Shift}");

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("never submits an empty or whitespace-only question", async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} isSubmitting={false} />);

    await userEvent.type(screen.getByRole("textbox"), "   ");
    await userEvent.keyboard("{Enter}");

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("disables the send button while a request is in flight, preventing duplicate submissions", () => {
    render(<QuestionInput onSubmit={vi.fn()} isSubmitting={true} />);

    expect(screen.getByRole("button", { name: "Send question" })).toBeDisabled();
  });

  it("has no detectable accessibility violations", async () => {
    const { container } = render(<QuestionInput onSubmit={vi.fn()} isSubmitting={false} />);

    const results = await axe(container);
    expect(results.violations).toEqual([]);
  });
});
