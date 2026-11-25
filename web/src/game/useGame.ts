import { useCallback, useEffect, useReducer, useState } from "react";
import { ApiError, requestMove } from "../api/client";
import { levelById } from "./levels";
import type { Cell } from "./rules";
import { type GameSettings, gameReducer, initialState, isEngineTurn, modeOf, sideSettings, sideToMove } from "./state";

export type EngineStatus =
  | { kind: "idle" }
  | { kind: "paused" }
  | { kind: "thinking"; since: number }
  | { kind: "error"; error: ApiError; retryAt?: number };

/** Pause between engine moves in engine vs engine, so a game can be followed. */
export const SPEEDS = [0, 500, 1500] as const;
export type Speed = (typeof SPEEDS)[number];

/**
 * Game state plus the engine loop: whenever an engine plays the side to
 * move, ask `POST /v1/move` for its move.  A new game or an undo aborts the
 * request in flight (and the reducer drops any reply that still arrives).
 *
 * Engine vs engine plays itself; it can be paused, advanced one move at a
 * time (`step`) and slowed down (`speed`).
 */
export function useGame(initial: GameSettings) {
  const [state, dispatch] = useReducer(gameReducer, initial, (s) => initialState(s, 1, Date.now(), false));
  const [engine, setEngine] = useState<EngineStatus>({ kind: "idle" });
  const [attempt, setAttempt] = useState(0);
  const [paused, setPaused] = useState(false);
  const [steps, setSteps] = useState(0); // moves allowed while paused
  const [speed, setSpeed] = useState<Speed>(500);

  const engineTurn = isEngineTurn(state);
  const { gameId, grid } = state;
  const ply = state.moves.length;
  const side = sideToMove(state);
  const presetId = sideSettings(state.settings, side).preset;
  const selfPlay = modeOf(state.settings) === "engine-engine";
  const held = selfPlay && paused && steps === 0;
  // Delay only between engine moves of a self-play game, never before the first.
  const delay = selfPlay && ply > 0 && !paused ? speed : 0;

  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` re-runs the request on retry; `grid` and `side` follow `ply`
  useEffect(() => {
    if (!engineTurn) {
      setEngine({ kind: "idle" });
      return;
    }
    if (held) {
      setEngine({ kind: "paused" });
      return;
    }
    const controller = new AbortController();
    const preset = levelById(presetId);
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    const timer = setTimeout(() => {
      setEngine({ kind: "thinking", since: Date.now() });
      requestMove({ grid, sideToMove: side, brain: preset.brain, overrides: preset.overrides }, controller.signal)
        .then((move) => {
          if (paused) setSteps((n) => Math.max(0, n - 1));
          dispatch({
            type: "play",
            gameId,
            ply,
            cell: move.cell,
            by: "engine",
            thinkMs: move.elapsedMs,
            at: Date.now(),
          });
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const error = err instanceof ApiError ? err : new ApiError("unavailable", undefined, String(err));
          // Rate limited: try again by itself once the server says it may.
          const wait = error.kind === "rate_limited" ? (error.retryAfterS ?? 10) : null;
          setEngine({ kind: "error", error, retryAt: wait === null ? undefined : Date.now() + wait * 1000 });
          if (wait !== null) retryTimer = setTimeout(() => setAttempt((n) => n + 1), wait * 1000);
        });
    }, delay);
    return () => {
      clearTimeout(timer);
      clearTimeout(retryTimer);
      controller.abort();
    };
  }, [engineTurn, gameId, ply, presetId, attempt, held, delay]);

  const play = useCallback(
    (cell: Cell) => dispatch({ type: "play", gameId, ply, cell, by: "human", at: Date.now() }),
    [gameId, ply],
  );
  const newGame = useCallback((settings: GameSettings) => {
    setPaused(false);
    setSteps(0);
    dispatch({ type: "new", settings, at: Date.now() });
  }, []);
  const undo = useCallback(() => dispatch({ type: "undo", at: Date.now() }), []);
  const resign = useCallback(() => dispatch({ type: "resign", at: Date.now() }), []);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const togglePause = useCallback(() => {
    setSteps(0);
    setPaused((p) => !p);
  }, []);
  const step = useCallback(() => setSteps((n) => (n === 0 ? 1 : n)), []);

  return { state, engine, play, newGame, undo, resign, retry, paused, togglePause, step, speed, setSpeed };
}
