import { AxiosError } from "axios";
import { describe, expect, it } from "vitest";

import { toAppError } from "@/lib/http";
import { AppError } from "@/types/errors";

function axiosErrorWithResponse(status: number, data: unknown): AxiosError {
  const error = new AxiosError("Request failed", undefined, undefined, undefined, {
    status,
    data,
    statusText: "",
    headers: {},
    // @ts-expect-error -- config is not needed for this test's assertions
    config: {},
  });
  return error;
}

describe("toAppError", () => {
  it("passes an existing AppError through unchanged", () => {
    const original = new AppError("already normalized", "validation");
    expect(toAppError(original, "fallback")).toBe(original);
  });

  it("classifies a timeout (ECONNABORTED) as kind 'timeout'", () => {
    const error = new AxiosError("timeout", "ECONNABORTED");
    const result = toAppError(error, "fallback");
    expect(result.kind).toBe("timeout");
  });

  it("classifies a response-less error as kind 'network'", () => {
    const error = new AxiosError("no response");
    const result = toAppError(error, "fallback");
    expect(result.kind).toBe("network");
  });

  it("classifies a 404 as kind 'not_found'", () => {
    const result = toAppError(axiosErrorWithResponse(404, {}), "fallback");
    expect(result.kind).toBe("not_found");
    expect(result.status).toBe(404);
  });

  it("classifies a 422 as kind 'validation'", () => {
    const result = toAppError(axiosErrorWithResponse(422, {}), "fallback");
    expect(result.kind).toBe("validation");
  });

  it("classifies a 500 as kind 'server'", () => {
    const result = toAppError(axiosErrorWithResponse(500, {}), "fallback");
    expect(result.kind).toBe("server");
  });

  it("prefers the backend's own detail message when present", () => {
    const result = toAppError(
      axiosErrorWithResponse(422, { detail: "specific reason" }),
      "fallback",
    );
    expect(result.message).toBe("specific reason");
  });

  it("falls back to the provided message when no detail is present", () => {
    const result = toAppError(axiosErrorWithResponse(500, {}), "fallback message");
    expect(result.message).toBe("fallback message");
  });

  it("classifies a completely unrecognized error as kind 'unknown'", () => {
    const result = toAppError(new Error("boom"), "fallback");
    expect(result.kind).toBe("unknown");
  });
});
