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
  /**
   * Difficulty level id (config.json) when `kind` is `engine`; kept for
   * humans so switching back remembers it.
   */
  readonly preset: string;
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
  /**
   * Time this move took, for the players' clocks: a human's from the start
   * of their turn, an engine's from its request (pauses and the delay
   * between engine moves do not count).
   */
  readonly spentMs: number;
}

export type Outcome =
  | { readonly kind: "win"; readonly winner: Player; readonly line: readonly Cell[] }
  | {
      readonly kind: "resign";
      readonly winner: Player;
      /** Time the resigning player spent on the turn they resigned in. */
      readonly spentMs: number;
    }
  | { readonly kind: "draw" };

export interface GameState {
  /** Increments on every new game, so late engine replies to an old game are dropped. */
  readonly gameId: number;
  readonly settings: GameSettings;
  readonly grid: Grid;
  readonly moves: readonly MoveRecord[];
  readonly outcome: Outcome | null;
  /** When the side to move started its turn (ms, `Date.now()` clock). */
  readonly turnStartedAt: number;
  /**
   * False until a game is started from the New game dialog: the page opens
   * on an empty board where nobody moves and no clock runs.
   */
  readonly started: boolean;
}

export type GameAction =
  | { type: "new"; settings: GameSettings; at: number }
  | { type: "play"; gameId: number; ply: number; cell: Cell; by: PlayerKind; thinkMs?: number; at: number }
  | { type: "undo"; at: number }
  | { type: "resign"; at: number };

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

export function initialState(
  settings: GameSettings = DEFAULT_SETTINGS,
  gameId = 1,
  at = Date.now(),
  started = true,
): GameState {
  return { gameId, settings, grid: emptyGrid(settings.size), moves: [], outcome: null, turnStartedAt: at, started };
}

/** Black always moves first. */
export function sideToMove(state: GameState): Player {
  return state.moves.length % 2 === 0 ? BLACK : WHITE;
}

/** Who plays the side to move, or null once the game is over. */
export function turnKind(state: GameState): PlayerKind | null {
  if (!state.started || state.outcome !== null) return null;
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
      return initialState(action.settings, state.gameId + 1, action.at);

    case "play": {
      // Stale reply (new game, undo) or illegal move: ignore.
      if (action.gameId !== state.gameId || action.ply !== state.moves.length) return state;
      if (!state.started || state.outcome !== null || !isLegal(state.grid, action.cell)) return state;
      const stone = sideToMove(state);
      // A move must come from whoever plays this colour.
      if (sideSettings(state.settings, stone).kind !== action.by) return state;
      const grid = placeStone(state.grid, action.cell, stone);
      const spentMs =
        action.by === "engine" && action.thinkMs !== undefined
          ? action.thinkMs
          : Math.max(0, action.at - state.turnStartedAt);
      const record: MoveRecord = { ...action.cell, stone, by: action.by, thinkMs: action.thinkMs, spentMs };
      const line = winningLine(grid, action.cell, state.settings.winLength);
      const outcome: Outcome | null = line
        ? { kind: "win", winner: stone, line }
        : isFull(grid)
          ? { kind: "draw" }
          : null;
      return { ...state, grid, moves: [...state.moves, record], outcome, turnStartedAt: action.at };
    }

    case "resign": {
      if (!canResign(state)) return state;
      const spentMs = Math.max(0, action.at - state.turnStartedAt);
      return { ...state, outcome: { kind: "resign", winner: opponent(sideToMove(state)), spentMs } };
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
      return { ...state, grid: replay(state.settings, moves), moves, outcome: null, turnStartedAt: action.at };
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
    const preset = typeof r.preset === "string" ? r.preset : "hard";
    const human: SideSettings = { kind: "human", preset };
    const engine: SideSettings = { kind: "engine", preset };
    return r.human === BLACK
      ? { size, winLength, black: human, white: engine }
      : { size, winLength, black: engine, white: human };
  }
  return { ...DEFAULT_SETTINGS, size, winLength };
}

export { opponent };

/**
 * A player's clock: the time they have spent in this game, counting up like
 * a chess clock.  It runs while it is their turn: a human's from the start
 * of the turn, an engine's only while its request is in flight
 * (`engineSince`), so pauses and errors do not count.
 */
export function clockMs(
  state: GameState,
  player: Player,
  now: number,
  engineSince: number | null,
): { readonly ms: number; readonly running: boolean } {
  let ms = state.moves.reduce((sum, m) => (m.stone === player ? sum + m.spentMs : sum), 0);
  const { outcome } = state;
  if (outcome?.kind === "resign" && outcome.winner !== player) ms += outcome.spentMs;
  if (turnKind(state) === null || sideToMove(state) !== player) return { ms, running: false };
  if (sideSettings(state.settings, player).kind === "human") {
    return { ms: ms + Math.max(0, now - state.turnStartedAt), running: true };
  }
  if (engineSince !== null) return { ms: ms + Math.max(0, now - engineSince), running: true };
  return { ms, running: false };
}

/** `m:ss`, or `h:mm:ss` from an hour. */
export function formatClock(ms: number): string {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = String(total % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}
