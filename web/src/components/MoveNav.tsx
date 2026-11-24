import { clsx } from "clsx";
import { t } from "../lib/i18n";
import { FirstIcon, LastIcon, NextIcon, PrevIcon } from "./icons";
import { Button } from "./ui";

/**
 * ⏮ ◀ ▶ ⏭ through the game.  `ply` is the number of moves shown (0 = empty
 * board); `total` the moves played.  Viewing an earlier position never
 * changes the game: it goes on, and ⏭ catches up with it.
 */
export function MoveNav({ ply, total, onGo }: { ply: number; total: number; onGo: (ply: number) => void }) {
  const live = ply === total;
  return (
    <div className="flex w-full items-center gap-2" role="toolbar" aria-label={t.navLabel}>
      <div className="flex gap-1">
        <Button
          icon
          onClick={() => onGo(0)}
          disabled={ply === 0}
          aria-label={t.navFirst}
          title={`${t.navFirst} (Home)`}
        >
          <FirstIcon />
        </Button>
        <Button
          icon
          onClick={() => onGo(ply - 1)}
          disabled={ply === 0}
          aria-label={t.navPrev}
          title={`${t.navPrev} (←)`}
        >
          <PrevIcon />
        </Button>
        <Button icon onClick={() => onGo(ply + 1)} disabled={live} aria-label={t.navNext} title={`${t.navNext} (→)`}>
          <NextIcon />
        </Button>
        <Button icon onClick={() => onGo(total)} disabled={live} aria-label={t.navLast} title={`${t.navLast} (End)`}>
          <LastIcon />
        </Button>
      </div>
      <span
        className={clsx("whitespace-nowrap text-sm tabular-nums", live ? "text-muted" : "font-semibold text-accent")}
        aria-live="polite"
      >
        {live ? t.navLive(total) : t.navViewing(ply, total)}
      </span>
    </div>
  );
}
