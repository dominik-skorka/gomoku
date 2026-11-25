import { clsx } from "clsx";
import { useEffect, useRef } from "react";
import { positionLabel } from "../game/rules";
import type { MoveRecord } from "../game/state";
import { t } from "../lib/i18n";
import { StoneIcon } from "./ui";

/**
 * The moves played.  Each one is a button that shows the position after it;
 * the move on view is highlighted.
 */
export function MoveList({
  moves,
  viewPly,
  onGo,
}: {
  moves: readonly MoveRecord[];
  /** Moves shown on the board (equals `moves.length` when live). */
  viewPly: number;
  onGo: (ply: number) => void;
}) {
  const list = useRef<HTMLOListElement>(null);
  const current = useRef<HTMLLIElement>(null);
  // Keep the move on view visible inside the list.  Only the list scrolls:
  // scrollIntoView() would also scroll the page, pulling a phone's screen
  // away from the board after every move.
  useEffect(() => {
    const box = list.current;
    const item = current.current;
    if (!box || !item) return;
    // Latest move: all the way down, padding included.
    if (viewPly === moves.length) {
      box.scrollTop = box.scrollHeight;
      return;
    }
    // The list is the items' offsetParent (position: relative), so offsetTop
    // is already measured from the top of the list's content.
    const top = item.offsetTop;
    if (top < box.scrollTop) box.scrollTop = top;
    else if (top + item.offsetHeight > box.scrollTop + box.clientHeight)
      box.scrollTop = top + item.offsetHeight - box.clientHeight;
  }, [viewPly, moves.length]);

  if (moves.length === 0) return null;
  return (
    <ol
      aria-label={t.moves}
      ref={list}
      className="relative grid max-h-64 grid-cols-2 gap-x-3 gap-y-1 overflow-y-auto p-0.5 text-sm tabular-nums lg:max-h-[40dvh]"
    >
      {moves.map((m, i) => (
        <li key={`${m.row}-${m.col}`} ref={i === viewPly - 1 ? current : undefined}>
          <button
            type="button"
            onClick={() => onGo(i + 1)}
            aria-current={i === viewPly - 1 ? "step" : undefined}
            className={clsx(
              "flex w-full items-center gap-2 rounded-lg px-2 py-1 text-left transition",
              "focus-visible:outline-2 focus-visible:outline-accent",
              i === viewPly - 1
                ? "bg-accent/15 ring-1 ring-accent ring-inset"
                : i % 2 === 0
                  ? "bg-black/[0.03] hover:bg-black/[0.06] dark:bg-white/[0.03] dark:hover:bg-white/[0.06]"
                  : "hover:bg-black/[0.04] dark:hover:bg-white/[0.04]",
              i >= viewPly && "opacity-45",
            )}
          >
            <span className="w-6 text-right text-xs text-muted">{i + 1}.</span>
            <StoneIcon stone={m.stone} className="size-3" />
            <span className="font-medium">{positionLabel(m)}</span>
            {m.thinkMs !== undefined && (
              <span className="ml-auto text-xs text-muted">{(m.thinkMs / 1000).toFixed(1)}s</span>
            )}
          </button>
        </li>
      ))}
    </ol>
  );
}
