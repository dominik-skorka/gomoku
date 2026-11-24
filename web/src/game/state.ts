import type { PresetId } from "./presets";
import {
  BLACK,
  type Cell,
  emptyGrid,
  type Grid,
  isFull,
  isLegal,
  opponent,
  type Player,
  placeStone,
  WHITE,
  winningLine,
} from "./rules";

/**
 * The whole game lives in the browser: a pure reducer over an immutable
 * state.  Restarting or scaling the API never interrupts a game, and the
 * state can be tested without a server.
 */

export interface GameSettings {
  readonly size: number;
  readonly winLength: number;
  readonly human: Player;
  readonly preset: PresetId;
}

export interface MoveRecord extends Cell {
  readonly stone: Player;
  readonly by: "human" | "engine";
  /** Round-trip time of the engine request, for engine moves. */
  readonly thinkMs?: number;
}

export type Outcome =
  | { readonly kind: "win"; readonly winner: Player; readonly line: readonly Cell[] }
  | { readonly kind: "draw" };

export interface GameState {
  /** Increments on every new game, so late engine replies to an old game are dropped. */
  readonly gameId: number;
  readonly settings: GameSettings;
  readonly grid: Grid;
  readonly moves: readonly MoveRecord[];
  readonly outcome: Outcome | null;
}

export type GameAction =
  | { type: "new"; settings: GameSettings }
  | { type: "play"; gameId: number; ply: number; cell: Cell; by: MoveRecord["by"]; thinkMs?: number }
  | { type: "undo" };

export const DEFAULT_SETTINGS: GameSettings = { size: 15, winLength: 5, human: BLACK, preset: "hard" };

export function initialState(settings: GameSettings = DEFAULT_SETTINGS, gameId = 1): GameState {
  return { gameId, settings, grid: emptyGrid(settings.size), moves: [], outcome: null };
}

/** Black always moves first. */
export function sideToMove(state: GameState): Player {
  return state.moves.length % 2 === 0 ? BLACK : WHITE;
}

export function isEngineTurn(state: GameState): boolean {
  return state.outcome === null && sideToMove(state) !== state.settings.human;
}

function replay(settings: GameSettings, moves: readonly MoveRecord[]): Grid {
  return moves.reduce((g, m) => placeStone(g, m, m.stone), emptyGrid(settings.size));
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "new":
      return initialState(action.settings, state.gameId + 1);

    case "play": {
      // Stale reply (new game, undo) or illegal move: ignore.
      if (action.gameId !== state.gameId || action.ply !== state.moves.length) return state;
      if (state.outcome !== null || !isLegal(state.grid, action.cell)) return state;
      const stone = sideToMove(state);
      if ((action.by === "human") !== (stone === state.settings.human)) return state;
      const grid = placeStone(state.grid, action.cell, stone);
      const record: MoveRecord = { ...action.cell, stone, by: action.by, thinkMs: action.thinkMs };
      const line = winningLine(grid, action.cell, state.settings.winLength);
      const outcome: Outcome | null = line
        ? { kind: "win", winner: stone, line }
        : isFull(grid)
          ? { kind: "draw" }
          : null;
      return { ...state, grid, moves: [...state.moves, record], outcome };
    }

    case "undo": {
      // Take back to the human's previous turn: drop the engine's reply (if
      // any) and the human's last move.
      const moves = [...state.moves];
      while (moves.length > 0 && moves.at(-1)?.by === "engine") moves.pop();
      if (moves.length === 0) return state;
      moves.pop();
      // Engine opened the game (human plays White): keep its first move.
      return { ...state, grid: replay(state.settings, moves), moves, outcome: null };
    }
  }
}

export function canUndo(state: GameState): boolean {
  return state.moves.some((m) => m.by === "human");
}

export { opponent };
