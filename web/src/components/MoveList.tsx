import { useEffect, useRef } from "react";
import { positionLabel } from "../game/rules";
import type { MoveRecord } from "../game/state";
import { t } from "../lib/i18n";
import { StoneIcon } from "./ui";

export function MoveList({ moves }: { moves: readonly MoveRecord[] }) {
  const end = useRef<HTMLLIElement>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on every new move
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [moves.length]);

  if (moves.length === 0) return <p className="text-sm text-muted">{t.noMoves}</p>;
  return (
    <ol className="grid max-h-64 grid-cols-2 gap-x-3 gap-y-1 overflow-y-auto pr-1 text-sm tabular-nums lg:max-h-[40dvh]">
      {moves.map((m, i) => (
        <li
          key={`${m.row}-${m.col}`}
          ref={i === moves.length - 1 ? end : undefined}
          className="flex items-center gap-2 rounded-lg px-2 py-1 odd:bg-black/[0.03] dark:odd:bg-white/[0.03]"
        >
          <span className="w-6 text-right text-xs text-muted">{i + 1}.</span>
          <StoneIcon stone={m.stone} className="size-3" />
          <span className="font-medium">{positionLabel(m)}</span>
          {m.thinkMs !== undefined && (
            <span className="ml-auto text-xs text-muted">{(m.thinkMs / 1000).toFixed(1)}s</span>
          )}
        </li>
      ))}
    </ol>
  );
}
