import { beforeEach, describe, expect, it } from "vitest";
import { defaultsOf, getPath, groupsOf, setPath } from "../components/LevelSettingsDialog";
import { config } from "../lib/config";
import { exportConfig, levelById, levelChanges, levels, resetAllLevels, setLevelChange } from "./levels";

describe("levels", () => {
  beforeEach(() => resetAllLevels());

  it("are the config's levels until this browser changes one", () => {
    expect(levels()).toEqual(config.levels);
    setLevelChange("easy", { brain: "mcts", overrides: { iterations: 500 } });
    expect(levelById("easy")).toMatchObject({ id: "easy", brain: "mcts", overrides: { iterations: 500 } });
    expect(JSON.parse(localStorage.getItem("gomoku.levels") ?? "{}")).toHaveProperty("easy");
    // The export carries the change, for config.json.
    expect(JSON.parse(exportConfig()).levels[0]).toMatchObject({ id: "easy", brain: "mcts" });
    setLevelChange("easy", null);
    expect(levelById("easy")).toEqual(config.levels[0]);
    expect(levelChanges()).toEqual({});
  });

  it("forget a change that ends where config.json is", () => {
    const hard = config.levels.find((l) => l.id === "hard");
    if (!hard) throw new Error("no hard level");
    setLevelChange("hard", { brain: hard.brain, overrides: { depth: 6 } });
    expect(levelChanges()).toHaveProperty("hard");
    setLevelChange("hard", { brain: hard.brain, overrides: { ...hard.overrides } });
    expect(levelChanges()).not.toHaveProperty("hard");
  });

  it("fall back to the default level for an unknown id", () => {
    expect(levelById("gone").id).toBe(config.defaultLevel);
  });
});

describe("settings form", () => {
  const schema = {
    properties: {
      depth: { type: "integer", minimum: 1, maximum: 10, default: 4 },
      lmr: { type: "boolean", default: false },
      evaluator: { type: "string", default: "pattern" },
      threats: { $ref: "#/$defs/Search", description: "Which threats." },
      fallback: { anyOf: [{ $ref: "#/$defs/AB" }, { type: "null" }], default: null },
    },
    $defs: {
      Search: { properties: { own: { $ref: "#/$defs/Attack" } } },
      Attack: { properties: { fours: { type: "boolean", default: true } } },
      AB: { properties: { depth: { type: "integer", default: 4 } } },
    },
  };

  it("lists flat settings and walks into nested groups", () => {
    const groups = groupsOf(schema);
    expect(groups.map((g) => [g.path.join("."), g.fields.map((f) => f.key)])).toEqual([
      ["", ["depth", "lmr", "evaluator"]],
      ["threats", []],
      ["threats.own", ["fours"]],
      ["fallback", ["depth"]],
    ]);
    expect(groups[1]?.description).toBe("Which threats.");
    expect(groups[0]?.fields[0]).toMatchObject({ type: "integer", min: 1, max: 10 });
  });

  it("builds the schema defaults, nested ones included", () => {
    expect(defaultsOf(schema)).toEqual({
      depth: 4,
      lmr: false,
      evaluator: "pattern",
      threats: { own: { fours: true } },
      fallback: { depth: 4 },
    });
  });

  it("reads and writes paths without touching the original", () => {
    const original = { threats: { own: { fours: true, threes: true } } };
    const changed = setPath(original, ["threats", "own", "fours"], false);
    expect(changed).toEqual({ threats: { own: { fours: false, threes: true } } });
    expect(original.threats.own.fours).toBe(true);
    expect(getPath(changed, ["threats", "own", "threes"])).toBe(true);
    expect(getPath(changed, ["nope", "x"])).toBeUndefined();
  });
});
