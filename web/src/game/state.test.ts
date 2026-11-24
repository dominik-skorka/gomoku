import { describe, expect, it } from "vitest";
import { BLACK, WHITE } from "./rules";
import {
  canResign,
  canUndo,
  clockMs,
  formatClock,
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

function play(s: GameState, row: number, col: number, by: PlayerKind, at = 0, thinkMs?: number): GameState {
  return gameReducer(s, { type: "play", gameId: s.gameId, ply: s.moves.length, cell: { row, col }, by, at, thinkMs });
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
    expect(gameReducer(s, { type: "play", gameId: 0, ply: 1, cell: { row: 0, col: 0 }, by: "engine", at: 0 })).toBe(s);
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
    s = gameReducer(s, { type: "undo", at: 0 });
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
    s = gameReducer(s, { type: "undo", at: 0 });
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
    expect(gameReducer(s, { type: "undo", at: 0 })).toBe(s);
  });

  it("resign: the side to move gives up, only on a human turn, and it is final", () => {
    let s = initialState(hve);
    s = play(s, 4, 4, "human");
    // Engine's turn: the human cannot resign now.
    expect(canResign(s)).toBe(false);
    expect(gameReducer(s, { type: "resign", at: 0 })).toBe(s);
    s = play(s, 3, 3, "engine");
    s = gameReducer(s, { type: "resign", at: 0 });
    expect(s.outcome).toMatchObject({ kind: "resign", winner: WHITE });
    expect(isEngineTurn(s)).toBe(false);
    expect(canUndo(s)).toBe(false);
    expect(gameReducer(s, { type: "undo", at: 0 })).toBe(s);
    // Engines never resign.
    const e = initialState(eve);
    expect(gameReducer(e, { type: "resign", at: 0 })).toBe(e);
  });

  it("human vs human: whoever is to move resigns", () => {
    let s = initialState(hvh);
    s = play(s, 4, 4, "human");
    s = gameReducer(s, { type: "resign", at: 0 });
    expect(s.outcome).toMatchObject({ kind: "resign", winner: BLACK });
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
    const n = gameReducer(s, { type: "new", settings: { ...hve, size: 15 }, at: 0 });
    expect(n.gameId).toBe(s.gameId + 1);
    expect(n.grid).toHaveLength(15);
  });
});

describe("clocks", () => {
  it("count each player's time like a chess clock", () => {
    let s = initialState(hve, 1, 1_000);
    // Black (human) thinks 4 s, then White (engine) answers in 1.5 s.
    expect(clockMs(s, BLACK, 3_000, null)).toEqual({ ms: 2_000, running: true });
    s = play(s, 4, 4, "human", 5_000);
    expect(clockMs(s, BLACK, 9_000, null)).toEqual({ ms: 4_000, running: false });
    // Engine clock runs only while its request is in flight.
    expect(clockMs(s, WHITE, 6_000, null)).toEqual({ ms: 0, running: false });
    expect(clockMs(s, WHITE, 6_000, 5_500)).toEqual({ ms: 500, running: true });
    s = play(s, 3, 3, "engine", 7_000, 1_500);
    expect(clockMs(s, WHITE, 9_000, null)).toEqual({ ms: 1_500, running: false });
    // Black's second turn started when White's reply arrived.
    expect(clockMs(s, BLACK, 9_000, null)).toEqual({ ms: 6_000, running: true });
  });

  it("stop at a resignation, counting the resigning turn, and roll back on undo", () => {
    let s = initialState(hve, 1, 0);
    s = play(s, 4, 4, "human", 2_000);
    s = play(s, 3, 3, "engine", 3_000, 1_000);
    s = gameReducer(s, { type: "resign", at: 8_000 });
    expect(clockMs(s, BLACK, 99_000, null)).toEqual({ ms: 7_000, running: false });
    let u = initialState(hvh, 1, 0);
    u = play(u, 4, 4, "human", 2_000);
    u = play(u, 3, 3, "human", 5_000);
    u = gameReducer(u, { type: "undo", at: 10_000 });
    expect(clockMs(u, WHITE, 11_000, null)).toEqual({ ms: 1_000, running: true });
  });

  it("formats m:ss and h:mm:ss", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(65_900)).toBe("1:05");
    expect(formatClock(3_725_000)).toBe("1:02:05");
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
