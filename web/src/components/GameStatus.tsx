import { AnimatePresence, motion } from "motion/react";
import type { GameState } from "../game/state";
import { sideToMove } from "../game/state";
import type { EngineStatus } from "../game/useGame";
import { t } from "../lib/i18n";
import { Button, StoneIcon } from "./ui";

function errorText(engine: Extract<EngineStatus, { kind: "error" }>): string {
  switch (engine.error.kind) {
    case "rate_limited":
      return t.errorRateLimited;
    case "unavailable":
      return t.errorUnavailable;
    case "rejected":
      return t.errorRejected;
  }
}

/** One line that always says what happens next.  Announced to screen readers. */
export function GameStatus({
  state,
  engine,
  onRetry,
}: {
  state: GameState;
  engine: EngineStatus;
  onRetry: () => void;
}) {
  const { outcome, settings } = state;
  let key: string;
  let content: React.ReactNode;

  if (outcome?.kind === "win") {
    key = "win";
    const won = outcome.winner === settings.human;
    content = (
      <span className={won ? "text-accent" : undefined}>
        <StoneIcon stone={outcome.winner} className="mr-2 align-[-2px]" />
        {won ? t.youWon : t.youLost}
      </span>
    );
  } else if (outcome?.kind === "draw") {
    key = "draw";
    content = t.draw;
  } else if (engine.kind === "error") {
    key = "error";
    content = (
      <span className="flex flex-wrap items-center gap-3">
        <span className="text-red-600 dark:text-red-400">{errorText(engine)}</span>
        <Button className="h-8 px-3" onClick={onRetry}>
          {t.retry}
        </Button>
      </span>
    );
  } else {
    const side = sideToMove(state);
    const mine = side === settings.human;
    key = mine ? "you" : "engine";
    content = (
      <span className="flex items-center gap-2">
        <StoneIcon stone={side} />
        {mine ? t.yourTurn : t.engineThinking}
        {!mine && (
          <span className="flex gap-1" aria-hidden="true">
            {[0, 1, 2].map((i) => (
              <motion.span
                key={i}
                className="size-1.5 rounded-full bg-current"
                animate={{ opacity: [0.2, 1, 0.2] }}
                transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, delay: i * 0.18 }}
              />
            ))}
          </span>
        )}
      </span>
    );
  }

  return (
    <div className="min-h-8 text-lg font-semibold" aria-live="polite">
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={key}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
        >
          {content}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
