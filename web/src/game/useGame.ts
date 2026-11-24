import { useCallback, useEffect, useReducer, useState } from "react";
import { ApiError, requestMove } from "../api/client";
import { presetById } from "./presets";
import type { Cell } from "./rules";
import { type GameSettings, gameReducer, initialState, isEngineTurn, sideToMove } from "./state";

export type EngineStatus = { kind: "idle" } | { kind: "thinking"; since: number } | { kind: "error"; error: ApiError };

/**
 * Game state plus the engine loop: whenever it is the engine's turn, ask
 * `POST /v1/move` for a move.  A new game or an undo aborts the request in
 * flight (and the reducer drops any reply that still arrives).
 */
export function useGame(initial: GameSettings) {
  const [state, dispatch] = useReducer(gameReducer, initial, (s) => initialState(s));
  const [engine, setEngine] = useState<EngineStatus>({ kind: "idle" });
  const [attempt, setAttempt] = useState(0);

  const engineTurn = isEngineTurn(state);
  const { gameId, grid } = state;
  const ply = state.moves.length;
  const { preset: presetId } = state.settings;
  const side = sideToMove(state);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `attempt` re-runs the request on retry
  useEffect(() => {
    if (!engineTurn) {
      setEngine({ kind: "idle" });
      return;
    }
    const controller = new AbortController();
    const preset = presetById(presetId);
    setEngine({ kind: "thinking", since: performance.now() });
    requestMove({ grid, sideToMove: side, brain: preset.brain, overrides: preset.overrides }, controller.signal)
      .then((move) => {
        dispatch({ type: "play", gameId, ply, cell: move.cell, by: "engine", thinkMs: move.elapsedMs });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const error = err instanceof ApiError ? err : new ApiError("unavailable", undefined, String(err));
        setEngine({ kind: "error", error });
      });
    return () => controller.abort();
  }, [engineTurn, gameId, ply, presetId, attempt]);

  const play = useCallback((cell: Cell) => dispatch({ type: "play", gameId, ply, cell, by: "human" }), [gameId, ply]);
  const newGame = useCallback((settings: GameSettings) => dispatch({ type: "new", settings }), []);
  const undo = useCallback(() => dispatch({ type: "undo" }), []);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return { state, engine, play, newGame, undo, retry };
}
