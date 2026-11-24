import { describe, expect, it } from "vitest";
import { apiUrl, baseOriginUrl, detectBasePath } from "./basePath";

describe("basePath", () => {
  it("returns empty base path for root and root SPA routes", () => {
    expect(detectBasePath("/")).toBe("");
    expect(detectBasePath("/playground")).toBe("");
    expect(detectBasePath("/playground/")).toBe("");
    expect(detectBasePath("/playground/chat-completions")).toBe("");
    expect(detectBasePath("/connect")).toBe("");
    expect(detectBasePath("/connect/")).toBe("");
  });

  it("extracts reverse-proxy subpaths across overview, playground, and connect", () => {
    expect(detectBasePath("/upstream/qwen3.8-27b-warp34/")).toBe(
      "/upstream/qwen3.8-27b-warp34",
    );
    expect(detectBasePath("/upstream/qwen3.8-27b-warp34")).toBe(
      "/upstream/qwen3.8-27b-warp34",
    );
    expect(detectBasePath("/upstream/qwen3.8-27b-warp34/playground")).toBe(
      "/upstream/qwen3.8-27b-warp34",
    );
    expect(
      detectBasePath("/upstream/qwen3.8-27b-warp34/playground/responses"),
    ).toBe("/upstream/qwen3.8-27b-warp34");
    expect(detectBasePath("/upstream/qwen3.8-27b-warp34/connect")).toBe(
      "/upstream/qwen3.8-27b-warp34",
    );
  });

  it("prefixes API URLs and origin URLs consistently", () => {
    expect(apiUrl("/api/metrics", "")).toBe("/api/metrics");
    expect(apiUrl("/api/metrics", "/upstream/model")).toBe(
      "/upstream/model/api/metrics",
    );
    expect(apiUrl("v1/models", "/upstream/model")).toBe(
      "/upstream/model/v1/models",
    );
    expect(baseOriginUrl("http://100.74.240.6:8080", "")).toBe(
      "http://100.74.240.6:8080",
    );
    expect(baseOriginUrl("http://100.74.240.6:8080", "/upstream/model")).toBe(
      "http://100.74.240.6:8080/upstream/model",
    );
  });
});
