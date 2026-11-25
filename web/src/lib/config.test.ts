import { describe, expect, it, vi } from "vitest";
import bundled from "../../public/config/config.json";
import { config, loadConfig, parseConfig } from "./config";

const good = {
  apiBase: "/api",
  defaultLevel: "a",
  levels: [
    { id: "a", brain: "negamax", overrides: { depth: 2 }, typicalSeconds: 0.5 },
    { id: "b", brain: "threat", typicalSeconds: 6 },
  ],
};

describe("parseConfig", () => {
  it("accepts the bundled config.json and a minimal one", () => {
    expect("config" in parseConfig(bundled)).toBe(true);
    const parsed = parseConfig(good);
    expect(parsed).toEqual({
      config: { ...good, levels: [good.levels[0], { ...good.levels[1], overrides: {} }] },
    });
  });

  it("names every problem", () => {
    const parsed = parseConfig({
      apiBase: "",
      defaultLevel: "zzz",
      levels: [
        { id: "a b", brain: "", typicalSeconds: -1, overrides: [] },
        { id: "x", brain: "b", typicalSeconds: 1 },
        { id: "x", brain: "b", typicalSeconds: 1 },
      ],
    });
    expect("errors" in parsed && parsed.errors).toEqual([
      "apiBase must be a non-empty string",
      "levels[0].id must be letters, digits, _ or -",
      "levels[0].brain must be a non-empty string",
      "levels[0].overrides must be an object",
      "levels[0].typicalSeconds must be a number >= 0",
      'levels[2].id "x" is used twice',
      "defaultLevel must be the id of one of the levels",
    ]);
    expect("errors" in parseConfig([])).toBe(true);
  });
});

describe("loadConfig", () => {
  it("uses the served file when valid, the bundled copy otherwise", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    await loadConfig(async () => new Response(JSON.stringify(good)));
    expect(config.defaultLevel).toBe("a");
    await loadConfig(async () => new Response(JSON.stringify({ apiBase: "/api" })));
    expect(config.defaultLevel).toBe(bundled.defaultLevel);
    await loadConfig(async () => new Response("", { status: 404 }));
    expect(config.levels).toHaveLength(bundled.levels.length);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});
