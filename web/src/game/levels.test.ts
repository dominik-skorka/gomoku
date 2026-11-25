import { beforeEach, describe, expect, it } from "vitest";
import { fieldsOf } from "../components/LevelSettingsDialog";
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

  it("fall back to the default level for an unknown id", () => {
    expect(levelById("gone").id).toBe(config.defaultLevel);
  });
});

describe("fieldsOf", () => {
  it("keeps flat settings, hides server paths, flags nested ones", () => {
    const { fields, hasNested } = fieldsOf({
      properties: {
        depth: { type: "integer", minimum: 1, maximum: 10, default: 4 },
        lmr: { type: "boolean", default: false },
        opening_book: { type: "string", default: "" },
        threats: { $ref: "#/$defs/X" },
      },
    });
    expect(fields.map((f) => f.key)).toEqual(["depth", "lmr"]);
    expect(fields[0]).toMatchObject({ type: "integer", min: 1, max: 10 });
    expect(hasNested).toBe(true);
  });
});
