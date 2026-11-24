import { clsx } from "clsx";
import { useEffect, useState } from "react";
import { BLACK, type Player } from "../game/rules";
import { clockMs, formatClock, type GameState, sideSettings, sideToMove, soleHuman } from "../game/state";
import type { EngineStatus } from "../game/useGame";
import { t } from "../lib/i18n";
import { errorText } from "../lib/status";
import { Button } from "./ui";

/** Current time, re-read every `ms` while `active`. */
function useNow(active: boolean, ms = 250): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setNow(Date.now());
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [active, ms]);
  return now;
}

function PersonIcon() {
  return (
    <svg viewBox="0 0 24 24" className="size-6" fill="currentColor" aria-hidden="true">
      <circle cx="12" cy="8.5" r="4" />
      <path d="M4.5 20a7.5 7.5 0 0 1 15 0z" />
    </svg>
  );
}

function EngineIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="size-6"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="6" y="6" width="12" height="12" rx="2.5" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="currentColor" />
      <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" />
    </svg>
  );
}

/**
 * Avatar frame.  A placeholder for now (a person or an engine on the
 * colour of the player's stones); an image can take its place later.
 */
function Avatar({ player, engine }: { player: Player; engine: boolean }) {
  return (
    <div
      data-avatar
      className={clsx(
        "grid size-11 shrink-0 place-items-center overflow-hidden rounded-xl ring-1 ring-line",
        player === BLACK
          ? "bg-[radial-gradient(circle_at_35%_30%,#4b5160,#1d2028_55%,#050608)] text-white/85"
          : "bg-[radial-gradient(circle_at_35%_30%,#fff,#e9ebef_70%,#c3c8d0)] text-slate-700",
      )}
    >
      {engine ? <EngineIcon /> : <PersonIcon />}
    </div>
  );
}

function Dots() {
  return (
    <span className="flex gap-1" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="svg-pulse size-1.5 rounded-full bg-current"
          style={{ animationDelay: `${i * 180}ms` }}
        />
      ))}
    </span>
  );
}

function Countdown({ at }: { at: number }) {
  const now = useNow(true, 500);
  return <span className="text-xs text-muted">{t.retryingIn(Math.max(0, Math.ceil((at - now) / 1000)))}</span>;
}

/**
 * One player, chess-style: avatar, name, what they are doing, and a clock
 * counting the time they have spent.  White's bar sits above the board,
 * Black's below.
 */
export function PlayerBar({
  player,
  state,
  engine,
  onRetry,
}: {
  player: Player;
  state: GameState;
  engine: EngineStatus;
  onRetry: () => void;
}) {
  const side = sideSettings(state.settings, player);
  const isEngine = side.kind === "engine";
  const toMove = state.outcome === null && sideToMove(state) === player;
  const engineSince = toMove && isEngine && engine.kind === "thinking" ? engine.since : null;
  const running = toMove && (!isEngine || engineSince !== null);
  const now = useNow(running);
  const clock = clockMs(state, player, now, engineSince);

  const you = soleHuman(state.settings);
  const name = isEngine ? `${t.engineKind} · ${t.presets[side.preset][0]}` : you === player ? t.youName : t.human;

  const { outcome } = state;
  let badge: React.ReactNode = null;
  if (outcome && outcome.kind !== "draw" && outcome.winner === player) {
    badge = <span className="rounded-full bg-accent px-2 py-0.5 text-xs font-bold text-accent-fg">{t.winsBadge}</span>;
  } else if (outcome?.kind === "resign") {
    badge = <span className="text-xs font-semibold text-muted">{t.resignedBadge}</span>;
  } else if (outcome?.kind === "draw") {
    badge = <span className="text-xs font-semibold text-muted">{t.drawBadge}</span>;
  } else if (toMove && engine.kind === "error") {
    badge = (
      <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-xs font-semibold text-red-600 dark:text-red-400">{errorText(engine)}</span>
        {engine.retryAt !== undefined && <Countdown at={engine.retryAt} />}
        <Button className="h-7 px-2.5 text-xs" onClick={onRetry}>
          {t.retry}
        </Button>
      </span>
    );
  } else if (toMove && isEngine && engine.kind === "paused") {
    badge = <span className="text-xs font-semibold text-muted">{t.paused}</span>;
  } else if (toMove && isEngine) {
    badge = <Dots />;
  }

  return (
    <div
      data-testid={`player-${player === BLACK ? "black" : "white"}`}
      data-active={toMove || undefined}
      className={clsx(
        "flex w-full items-center gap-3 rounded-2xl border px-2.5 py-2 transition-colors",
        toMove ? "border-accent/60 bg-surface-strong shadow-sm" : "border-transparent bg-surface/60",
      )}
    >
      <Avatar player={player} engine={isEngine} />
      <div className="flex min-h-11 min-w-0 flex-1 flex-col justify-center gap-0.5">
        <span className="truncate text-sm font-semibold">{name}</span>
        {badge && <div className="text-muted">{badge}</div>}
      </div>
      <span
        role="timer"
        data-testid="clock"
        aria-label={`${t.clock}: ${formatClock(clock.ms)}`}
        className={clsx(
          "rounded-lg px-2.5 py-1.5 font-mono text-lg font-semibold tabular-nums transition-colors",
          running ? "bg-fg text-bg" : "bg-black/5 text-muted dark:bg-white/10",
        )}
      >
        {formatClock(clock.ms)}
      </span>
    </div>
  );
}
