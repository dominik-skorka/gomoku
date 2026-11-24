import { describe, expect, it } from "vitest";
import { BLACK, WHITE } from "./rules";
import { canUndo, gameReducer, initialState, isEngineTurn, sideToMove } from "./state";

const settings = { size: 9, winLength: 5, human: BLACK, preset: "hard" } as const;

describe("gameReducer", () => {
  it("alternates turns and ignores out-of-turn or stale moves", () => {
    let s = initialState(settings);
    expect(isEngineTurn(s)).toBe(false);
    s = gameReducer(s, { type: "play", gameId: s.gameId, ply: 0, cell: { row: 4, col: 4 }, by: "human" });
    expect(sideToMove(s)).toBe(WHITE);
    expect(isEngineTurn(s)).toBe(true);
    // Human cannot move for the engine.
    expect(gameReducer(s, { type: "play", gameId: s.gameId, ply: 1, cell: { row: 0, col: 0 }, by: "human" })).toBe(s);
    // Reply to a previous game is dropped.
    expect(gameReducer(s, { type: "play", gameId: 0, ply: 1, cell: { row: 0, col: 0 }, by: "engine" })).toBe(s);
    // Occupied cell is rejected.
    expect(gameReducer(s, { type: "play", gameId: s.gameId, ply: 1, cell: { row: 4, col: 4 }, by: "engine" })).toBe(s);
  });

  it("ends the game on five in a row and blocks further moves", () => {
    let s = initialState(settings);
    for (let i = 0; i < 5; i++) {
      s = gameReducer(s, {
        type: "play",
        gameId: s.gameId,
        ply: s.moves.length,
        cell: { row: 0, col: i },
        by: "human",
      });
      if (i < 4) {
        s = gameReducer(s, {
          type: "play",
          gameId: s.gameId,
          ply: s.moves.length,
          cell: { row: 8, col: i },
          by: "engine",
        });
      }
    }
    expect(s.outcome).toMatchObject({ kind: "win", winner: BLACK });
    expect(isEngineTurn(s)).toBe(false);
  });

  it("undoes back to the human's previous turn, keeping the engine's opening", () => {
    let s = initialState({ ...settings, human: WHITE });
    s = gameReducer(s, { type: "play", gameId: s.gameId, ply: 0, cell: { row: 4, col: 4 }, by: "engine" });
    expect(canUndo(s)).toBe(false);
    s = gameReducer(s, { type: "play", gameId: s.gameId, ply: 1, cell: { row: 3, col: 3 }, by: "human" });
    s = gameReducer(s, { type: "play", gameId: s.gameId, ply: 2, cell: { row: 5, col: 5 }, by: "engine" });
    s = gameReducer(s, { type: "undo" });
    expect(s.moves).toHaveLength(1);
    expect(s.grid[3]?.[3]).toBe(0);
    expect(sideToMove(s)).toBe(WHITE);
  });

  it("starts a new game with a fresh id", () => {
    const s = initialState(settings);
    const n = gameReducer(s, { type: "new", settings: { ...settings, size: 15 } });
    expect(n.gameId).toBe(s.gameId + 1);
    expect(n.grid).toHaveLength(15);
  });
});
