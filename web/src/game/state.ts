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
 *
 * Each colour is played by a human (moves come from the board) or by an
 * engine (moves come from `POST /v1/move` at a difficulty preset), so one
 * model covers human vs engine, human vs human and engine vs engine.
 */

export type PlayerKind = "human" | "engine";

export interface SideSettings {
  readonly kind: PlayerKind;
  /** Difficulty when `kind` is `engine`; kept for humans so switching back remembers it. */
  readonly preset: PresetId;
}

export interface GameSettings {
  readonly size: number;
  readonly winLength: number;
  readonly black: SideSettings;
  readonly white: SideSettings;
}

export interface MoveRecord extends Cell {
  readonly stone: Player;
  readonly by: PlayerKind;
  /** Round-trip time of the engine request, for engine moves. */
  readonly thinkMs?: number;
}

export type Outcome =
  | { readonly kind: "win"; readonly winner: Player; readonly line: readonly Cell[] }
  | { readonly kind: "resign"; readonly winner: Player }
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
  | { type: "play"; gameId: number; ply: number; cell: Cell; by: PlayerKind; thinkMs?: number }
  | { type: "undo" }
  | { type: "resign" };

export const DEFAULT_SETTINGS: GameSettings = {
  size: 15,
  winLength: 5,
  black: { kind: "human", preset: "hard" },
  white: { kind: "engine", preset: "hard" },
};

export type Mode = "human-engine" | "human-human" | "engine-engine";

export function modeOf(settings: GameSettings): Mode {
  const humans = [settings.black, settings.white].filter((s) => s.kind === "human").length;
  return humans === 2 ? "human-human" : humans === 1 ? "human-engine" : "engine-engine";
}

export function sideSettings(settings: GameSettings, player: Player): SideSettings {
  return player === BLACK ? settings.black : settings.white;
}

/** The colour the only human plays, in human vs engine; otherwise null. */
export function soleHuman(settings: GameSettings): Player | null {
  if (modeOf(settings) !== "human-engine") return null;
  return settings.black.kind === "human" ? BLACK : WHITE;
}

export function initialState(settings: GameSettings = DEFAULT_SETTINGS, gameId = 1): GameState {
  return { gameId, settings, grid: emptyGrid(settings.size), moves: [], outcome: null };
}

/** Black always moves first. */
export function sideToMove(state: GameState): Player {
  return state.moves.length % 2 === 0 ? BLACK : WHITE;
}

/** Who plays the side to move, or null once the game is over. */
export function turnKind(state: GameState): PlayerKind | null {
  if (state.outcome !== null) return null;
  return sideSettings(state.settings, sideToMove(state)).kind;
}

export function isEngineTurn(state: GameState): boolean {
  return turnKind(state) === "engine";
}

export function isHumanTurn(state: GameState): boolean {
  return turnKind(state) === "human";
}

/** The grid after `moves`, from an empty board. */
export function replay(settings: GameSettings, moves: readonly MoveRecord[]): Grid {
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
      // A move must come from whoever plays this colour.
      if (sideSettings(state.settings, stone).kind !== action.by) return state;
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

    case "resign": {
      if (!canResign(state)) return state;
      return { ...state, outcome: { kind: "resign", winner: opponent(sideToMove(state)) } };
    }

    case "undo": {
      // Take back to the previous human turn: drop engine replies, then the
      // last human move.  Human vs human: one move.  Human vs engine: the
      // engine's reply and the human's move.  Engine vs engine: nothing.
      if (!canUndo(state)) return state;
      const moves = [...state.moves];
      while (moves.length > 0) {
        const m = moves.pop();
        if (m?.by === "human") break;
      }
      return { ...state, grid: replay(state.settings, moves), moves, outcome: null };
    }
  }
}

export function canUndo(state: GameState): boolean {
  // A resignation is final; a finished five-in-a-row can still be taken back.
  return state.outcome?.kind !== "resign" && state.moves.some((m) => m.by === "human");
}

/** Only a human, on their own turn, while the game is on. */
export function canResign(state: GameState): boolean {
  return isHumanTurn(state);
}

/**
 * Settings saved by earlier versions (`{ human, preset }`) or partly
 * missing: rebuilt into the current shape, falling back to the defaults.
 */
export function migrateSettings(raw: unknown): GameSettings {
  if (typeof raw !== "object" || raw === null) return DEFAULT_SETTINGS;
  const r = raw as Record<string, unknown>;
  const size = typeof r.size === "number" ? r.size : DEFAULT_SETTINGS.size;
  const winLength = typeof r.winLength === "number" ? r.winLength : DEFAULT_SETTINGS.winLength;
  const isSide = (v: unknown): v is SideSettings =>
    typeof v === "object" &&
    v !== null &&
    ((v as SideSettings).kind === "human" || (v as SideSettings).kind === "engine") &&
    typeof (v as SideSettings).preset === "string";
  if (isSide(r.black) && isSide(r.white)) return { size, winLength, black: r.black, white: r.white };
  if (r.human === BLACK || r.human === WHITE) {
    const preset = (typeof r.preset === "string" ? r.preset : "hard") as PresetId;
    const human: SideSettings = { kind: "human", preset };
    const engine: SideSettings = { kind: "engine", preset };
    return r.human === BLACK
      ? { size, winLength, black: human, white: engine }
      : { size, winLength, black: engine, white: human };
  }
  return { ...DEFAULT_SETTINGS, size, winLength };
}

export { opponent };
