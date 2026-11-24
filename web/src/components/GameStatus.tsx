import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { BLACK, type Player } from "../game/rules";
import type { GameState } from "../game/state";
import { sideToMove, soleHuman, turnKind } from "../game/state";
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

/** Seconds left until `at`, ticking once a second. */
function Countdown({ at }: { at: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span className="text-sm font-normal text-muted">{t.retryingIn(Math.max(0, Math.ceil((at - now) / 1000)))}</span>
  );
}

const colour = (p: Player) => (p === BLACK ? t.black : t.white);

function ThinkingDots() {
  return (
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
  );
}

/**
 * One line that always says what happens next.  Announced to screen readers.
 * Against one engine it speaks to "you"; otherwise it names the colours.
 */
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
  const you = soleHuman(settings);
  let key: string;
  let content: React.ReactNode;

  if (outcome?.kind === "win") {
    key = "win";
    const won = you === null || outcome.winner === you;
    const text = you === null ? t.wins(colour(outcome.winner)) : won ? t.youWon : t.youLost;
    content = (
      <span className={won ? "text-accent" : undefined}>
        <StoneIcon stone={outcome.winner} className="mr-2 align-[-2px]" />
        {text}
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
        {engine.retryAt !== undefined && <Countdown at={engine.retryAt} />}
        <Button className="h-8 px-3" onClick={onRetry}>
          {t.retry}
        </Button>
      </span>
    );
  } else {
    const side = sideToMove(state);
    const kind = turnKind(state);
    let text: string;
    if (kind === "human") text = you !== null ? t.yourTurn : t.toMove(colour(side));
    else if (engine.kind === "paused") text = t.paused;
    else text = you !== null ? t.engineThinking : t.isThinking(colour(side));
    key = `${kind}-${side}-${engine.kind}`;
    content = (
      <span className="flex items-center gap-2">
        <StoneIcon stone={side} />
        {text}
        {kind === "engine" && engine.kind !== "paused" && <ThinkingDots />}
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
