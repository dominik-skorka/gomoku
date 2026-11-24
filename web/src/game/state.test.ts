import { describe, expect, it } from "vitest";
import { BLACK, WHITE } from "./rules";
import {
  canResign,
  canUndo,
  type GameSettings,
  type GameState,
  gameReducer,
  initialState,
  isEngineTurn,
  isHumanTurn,
  migrateSettings,
  modeOf,
  type PlayerKind,
  replay,
  sideToMove,
} from "./state";

const human = { kind: "human", preset: "hard" } as const;
const engine = { kind: "engine", preset: "hard" } as const;
const hve: GameSettings = { size: 9, winLength: 5, black: human, white: engine };
const hvh: GameSettings = { ...hve, white: human };
const eve: GameSettings = { ...hve, black: engine };

function play(s: GameState, row: number, col: number, by: PlayerKind): GameState {
  return gameReducer(s, { type: "play", gameId: s.gameId, ply: s.moves.length, cell: { row, col }, by });
}

describe("gameReducer", () => {
  it("knows the three modes", () => {
    expect(modeOf(hve)).toBe("human-engine");
    expect(modeOf(hvh)).toBe("human-human");
    expect(modeOf(eve)).toBe("engine-engine");
  });

  it("human vs engine: alternates and ignores out-of-turn or stale moves", () => {
    let s = initialState(hve);
    expect(isHumanTurn(s)).toBe(true);
    s = play(s, 4, 4, "human");
    expect(sideToMove(s)).toBe(WHITE);
    expect(isEngineTurn(s)).toBe(true);
    // Human cannot move for the engine.
    expect(play(s, 0, 0, "human")).toBe(s);
    // Reply to a previous game is dropped.
    expect(gameReducer(s, { type: "play", gameId: 0, ply: 1, cell: { row: 0, col: 0 }, by: "engine" })).toBe(s);
    // Occupied cell is rejected.
    expect(play(s, 4, 4, "engine")).toBe(s);
  });

  it("ends the game on five in a row and blocks further moves", () => {
    let s = initialState(hve);
    for (let i = 0; i < 5; i++) {
      s = play(s, 0, i, "human");
      if (i < 4) s = play(s, 8, i, "engine");
    }
    expect(s.outcome).toMatchObject({ kind: "win", winner: BLACK });
    expect(isEngineTurn(s)).toBe(false);
    expect(isHumanTurn(s)).toBe(false);
  });

  it("human vs engine: undo drops the engine reply and the human move, keeping the engine's opening", () => {
    let s = initialState({ ...hve, black: engine, white: human });
    s = play(s, 4, 4, "engine");
    expect(canUndo(s)).toBe(false);
    s = play(s, 3, 3, "human");
    s = play(s, 5, 5, "engine");
    s = gameReducer(s, { type: "undo" });
    expect(s.moves).toHaveLength(1);
    expect(s.grid[3]?.[3]).toBe(0);
    expect(sideToMove(s)).toBe(WHITE);
  });

  it("human vs human: both colours move by hand and undo takes back one move", () => {
    let s = initialState(hvh);
    s = play(s, 4, 4, "human");
    expect(isHumanTurn(s)).toBe(true);
    expect(play(s, 3, 3, "engine")).toBe(s);
    s = play(s, 3, 3, "human");
    s = gameReducer(s, { type: "undo" });
    expect(s.moves).toHaveLength(1);
    expect(sideToMove(s)).toBe(WHITE);
  });

  it("engine vs engine: only engines move and undo does nothing", () => {
    let s = initialState(eve);
    expect(play(s, 4, 4, "human")).toBe(s);
    s = play(s, 4, 4, "engine");
    s = play(s, 3, 3, "engine");
    expect(isEngineTurn(s)).toBe(true);
    expect(canUndo(s)).toBe(false);
    expect(gameReducer(s, { type: "undo" })).toBe(s);
  });

  it("resign: the side to move gives up, only on a human turn, and it is final", () => {
    let s = initialState(hve);
    s = play(s, 4, 4, "human");
    // Engine's turn: the human cannot resign now.
    expect(canResign(s)).toBe(false);
    expect(gameReducer(s, { type: "resign" })).toBe(s);
    s = play(s, 3, 3, "engine");
    s = gameReducer(s, { type: "resign" });
    expect(s.outcome).toEqual({ kind: "resign", winner: WHITE });
    expect(isEngineTurn(s)).toBe(false);
    expect(canUndo(s)).toBe(false);
    expect(gameReducer(s, { type: "undo" })).toBe(s);
    // Engines never resign.
    const e = initialState(eve);
    expect(gameReducer(e, { type: "resign" })).toBe(e);
  });

  it("human vs human: whoever is to move resigns", () => {
    let s = initialState(hvh);
    s = play(s, 4, 4, "human");
    s = gameReducer(s, { type: "resign" });
    expect(s.outcome).toEqual({ kind: "resign", winner: BLACK });
  });

  it("replays any prefix of the game", () => {
    let s = initialState(hvh);
    s = play(s, 4, 4, "human");
    s = play(s, 3, 3, "human");
    expect(replay(s.settings, s.moves.slice(0, 1))[3]?.[3]).toBe(0);
    expect(replay(s.settings, s.moves)).toEqual(s.grid);
  });

  it("starts a new game with a fresh id", () => {
    const s = initialState(hve);
    const n = gameReducer(s, { type: "new", settings: { ...hve, size: 15 } });
    expect(n.gameId).toBe(s.gameId + 1);
    expect(n.grid).toHaveLength(15);
  });
});

describe("migrateSettings", () => {
  it("converts the old { human, preset } shape", () => {
    expect(migrateSettings({ size: 13, winLength: 5, human: WHITE, preset: "easy" })).toEqual({
      size: 13,
      winLength: 5,
      black: { kind: "engine", preset: "easy" },
      white: { kind: "human", preset: "easy" },
    });
  });

  it("keeps the current shape and rejects garbage", () => {
    expect(migrateSettings(eve)).toEqual(eve);
    expect(migrateSettings("nope").black.kind).toBe("human");
    expect(migrateSettings({ black: { kind: "robot" } }).white.kind).toBe("engine");
  });
});
