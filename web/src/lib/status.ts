import { BLACK, type Player } from "../game/rules";
import { type GameState, sideToMove, soleHuman, turnKind } from "../game/state";
import type { EngineStatus } from "../game/useGame";
import { t } from "./i18n";

export const colourName = (p: Player) => (p === BLACK ? t.black : t.white);

export function errorText(engine: Extract<EngineStatus, { kind: "error" }>): string {
  switch (engine.error.kind) {
    case "rate_limited":
      return t.errorRateLimited;
    case "unavailable":
      return t.errorUnavailable;
    case "rejected":
      return t.errorRejected;
  }
}

/**
 * One sentence describing the game, for screen readers (the player bars
 * show the same visually).  Against one engine it speaks to "you".
 */
export function statusText(state: GameState, engine: EngineStatus): string {
  const { outcome, settings } = state;
  if (!state.started) return t.notStarted;
  const you = soleHuman(settings);
  if (outcome?.kind === "win") {
    if (you === null) return t.wins(colourName(outcome.winner));
    return outcome.winner === you ? t.youWon : t.youLost;
  }
  if (outcome?.kind === "resign") {
    const loser: Player = outcome.winner === BLACK ? 2 : BLACK;
    if (you === null) return t.resigned(colourName(loser));
    return outcome.winner === you ? t.youWon : t.youResigned;
  }
  if (outcome?.kind === "draw") return t.draw;
  if (engine.kind === "error") return errorText(engine);
  const side = sideToMove(state);
  if (turnKind(state) === "human") return you !== null ? t.yourTurn : t.toMove(colourName(side));
  if (engine.kind === "paused") return t.paused;
  return you !== null ? t.engineThinking : t.isThinking(colourName(side));
}
