import { describe, expect, it, vi } from "vitest";
import en from "../../public/locales/en.json";
import pl from "../../public/locales/pl.json";
import { compile, fill, loadLocale, t } from "./i18n";

describe("i18n", () => {
  it("has the same keys in every locale", () => {
    expect(Object.keys(pl).sort()).toEqual(Object.keys(en).sort());
    expect(Object.keys(pl.levels).sort()).toEqual(Object.keys(en.levels).sort());
  });

  it("fills templates and level texts", () => {
    expect(fill("{a} and {b}", { a: 1, b: "x" })).toBe("1 and x");
    expect(fill("{missing}", {})).toBe("{missing}");
    const s = compile(en);
    expect(s.toMove("Black")).toBe("Black to move");
    expect(s.navViewing(3, 9)).toBe("Viewing 3/9");
    expect(s.seconds(0.5)).toBe("< 1 s");
    expect(s.seconds(6)).toBe("~6 s");
    expect(s.level("easy").label).toBe("Easy");
    // A level the locale does not know shows its id.
    expect(s.level("custom-7")).toEqual({ label: "custom-7", description: "" });
  });

  it("loads a locale, filling gaps from English", async () => {
    await loadLocale(
      "pl",
      async () => new Response(JSON.stringify({ newGame: "Nowa gra", levels: { easy: { label: "Łatwy" } } })),
    );
    expect(t.newGame).toBe("Nowa gra");
    expect(t.undo).toBe("Undo");
    expect(t.level("easy").label).toBe("Łatwy");
    expect(t.level("hard").label).toBe("Hard");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    await loadLocale("pl", async () => new Response("", { status: 500 }));
    expect(t.newGame).toBe("New game");
    warn.mockRestore();
  });
});
