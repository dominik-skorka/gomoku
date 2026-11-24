import { t } from "../lib/i18n";
import { FirstIcon, LastIcon, NextIcon, PrevIcon } from "./icons";
import { Button } from "./ui";

/**
 * ⏮ ◀ ▶ ⏭ through the game.  `ply` is the number of moves shown; `total`
 * the moves played.  ⏮ goes to the first move (not the empty board before
 * it).  Viewing an earlier position never changes the game: it goes on, and
 * ⏭ catches up with it.
 */
/** The earliest position navigation shows: after the first move. */
export const FIRST = 1;

export function MoveNav({ ply, total, onGo }: { ply: number; total: number; onGo: (ply: number) => void }) {
  const live = ply === total;
  const atFirst = ply <= FIRST;
  return (
    <div className="flex w-full items-center gap-2" role="toolbar" aria-label={t.navLabel}>
      <div className="flex gap-1">
        <Button
          icon
          onClick={() => onGo(FIRST)}
          disabled={atFirst}
          aria-label={t.navFirst}
          title={`${t.navFirst} (Home)`}
        >
          <FirstIcon />
        </Button>
        <Button icon onClick={() => onGo(ply - 1)} disabled={atFirst} aria-label={t.navPrev} title={`${t.navPrev} (←)`}>
          <PrevIcon />
        </Button>
        <Button icon onClick={() => onGo(ply + 1)} disabled={live} aria-label={t.navNext} title={`${t.navNext} (→)`}>
          <NextIcon />
        </Button>
        <Button icon onClick={() => onGo(total)} disabled={live} aria-label={t.navLast} title={`${t.navLast} (End)`}>
          <LastIcon />
        </Button>
      </div>
      {/* No visible label (the highlighted move in the list says where you are); screen readers still hear it. */}
      <span className="sr-only" aria-live="polite">
        {live ? t.navLive(total) : t.navViewing(ply, total)}
      </span>
    </div>
  );
}
