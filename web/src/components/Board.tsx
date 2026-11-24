import { type KeyboardEvent, type PointerEvent, useCallback, useId, useMemo, useRef, useState } from "react";
import { BLACK, type Cell, EMPTY, type Grid, isLegal, type Player, positionLabel } from "../game/rules";
import { t } from "../lib/i18n";

interface BoardProps {
  grid: Grid;
  /** Colour the human would place now, or null when input is disabled. */
  turn: Player | null;
  last: Cell | null;
  winLine: readonly Cell[] | null;
  hint: Cell | null;
  onPlay: (cell: Cell) => void;
}

const PAD = 1; // board margin, in cell units (room for coordinates)

const R = 0.45; // stone radius

function starPoints(size: number): Cell[] {
  if (size < 9) return [];
  const e = size >= 13 ? 3 : 2;
  const m = Math.floor(size / 2);
  const pts = [e, size - 1 - e];
  const out = pts.flatMap((row) => pts.map((col) => ({ row, col })));
  if (size % 2 === 1) out.push({ row: m, col: m });
  return out;
}

/**
 * The board as one SVG: crisp at any size, one pointer handler instead of
 * hundreds of hit targets.  Mouse: hover shows a ghost stone, click places.
 * Touch: one tap places (a two-tap confirmation was unreliable: iOS Safari
 * swallows a quick second tap as a double-tap gesture); Undo fixes a
 * mis-tap.  Keyboard: arrows + Enter.
 *
 * Drawn with plain shapes only: no SVG filters (WebKit failed to repaint a
 * filtered group of stones as stones were added) and no transforms.
 */
export function Board({ grid, turn, last, winLine, hint, onPlay }: BoardProps) {
  const size = grid.length;
  const span = size - 1 + PAD * 2;
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const [cursor, setCursor] = useState<Cell>({ row: Math.floor(size / 2), col: Math.floor(size / 2) });
  const [keyboard, setKeyboard] = useState(false);
  const uid = useId();

  const stones = useMemo(() => {
    const out: { row: number; col: number; stone: Player }[] = [];
    grid.forEach((row, r) => {
      row.forEach((s, c) => {
        if (s !== EMPTY) out.push({ row: r, col: c, stone: s });
      });
    });
    return out;
  }, [grid]);

  const cellAt = useCallback(
    (e: PointerEvent<SVGSVGElement>): Cell | null => {
      const svg = svgRef.current;
      const ctm = svg?.getScreenCTM();
      if (!svg || !ctm) return null;
      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
      const col = Math.round(p.x - PAD);
      const row = Math.round(p.y - PAD);
      if (row < 0 || col < 0 || row >= size || col >= size) return null;
      // Only near an intersection, so the gaps between points stay "dead".
      if (Math.hypot(p.x - PAD - col, p.y - PAD - row) > 0.5) return null;
      return { row, col };
    },
    [size],
  );

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (e.pointerType !== "mouse") return;
    setKeyboard(false);
    const cell = cellAt(e);
    setHover(cell && turn && isLegal(grid, cell) ? cell : null);
  };

  const onPointerUp = (e: PointerEvent<SVGSVGElement>) => {
    setKeyboard(false);
    if (!turn) return;
    const cell = cellAt(e);
    if (!cell || !isLegal(grid, cell)) return;
    setHover(null);
    onPlay(cell);
  };

  const onKeyDown = (e: KeyboardEvent<SVGSVGElement>) => {
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    const d = moves[e.key];
    if (d) {
      e.preventDefault();
      setKeyboard(true);
      setCursor((c) => ({
        row: Math.min(size - 1, Math.max(0, c.row + d[0])),
        col: Math.min(size - 1, Math.max(0, c.col + d[1])),
      }));
    } else if ((e.key === "Enter" || e.key === " ") && turn && isLegal(grid, cursor)) {
      e.preventDefault();
      onPlay(cursor);
    }
  };

  const ghost = hover;
  const x = (c: number) => PAD + c;
  const fontSize = 0.34;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${span} ${span}`}
      className="block h-full w-full touch-manipulation select-none outline-none [-webkit-touch-callout:none]"
      role="application"
      aria-label={`${t.boardLabel} ${size}×${size}`}
      // biome-ignore lint/a11y/noNoninteractiveTabindex: the board is an interactive widget (role=application)
      tabIndex={0}
      onPointerMove={onPointerMove}
      onPointerLeave={() => setHover(null)}
      onPointerUp={onPointerUp}
      onKeyDown={onKeyDown}
      onFocus={(e) => setKeyboard(e.currentTarget.matches(":focus-visible"))}
      onBlur={() => setKeyboard(false)}
      style={{ cursor: turn ? "pointer" : "default" }}
    >
      <defs>
        <linearGradient id={`${uid}-wood`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--board-a)" />
          <stop offset="1" stopColor="var(--board-b)" />
        </linearGradient>
        <radialGradient id={`${uid}-black`} cx="35%" cy="30%" r="75%">
          <stop offset="0" stopColor="#6b7180" />
          <stop offset="0.45" stopColor="#1d2028" />
          <stop offset="1" stopColor="#050608" />
        </radialGradient>
        <radialGradient id={`${uid}-white`} cx="35%" cy="30%" r="80%">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.7" stopColor="#e9ebef" />
          <stop offset="1" stopColor="#b9bec8" />
        </radialGradient>
        <radialGradient id={`${uid}-win`} cx="35%" cy="30%" r="80%">
          <stop offset="0" stopColor="#dbeafe" />
          <stop offset="0.45" stopColor="#3b82f6" />
          <stop offset="1" stopColor="#1e3a8a" />
        </radialGradient>
      </defs>

      <rect x="0" y="0" width={span} height={span} rx="0.35" fill={`url(#${uid}-wood)`} />

      <g stroke="var(--grid)" strokeWidth="0.035" strokeLinecap="square">
        {Array.from({ length: size }, (_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: static grid lines
          <g key={i}>
            <line x1={x(0)} y1={x(i)} x2={x(size - 1)} y2={x(i)} />
            <line x1={x(i)} y1={x(0)} x2={x(i)} y2={x(size - 1)} />
          </g>
        ))}
      </g>
      {starPoints(size).map((p) => (
        <circle key={`${p.row}-${p.col}`} cx={x(p.col)} cy={x(p.row)} r="0.09" fill="var(--grid)" />
      ))}

      <g fill="var(--label)" fontSize={fontSize} fontWeight={500} textAnchor="middle">
        {Array.from({ length: size }, (_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: static coordinates
          <g key={i}>
            <text x={x(i)} y={PAD / 2 + fontSize / 3}>
              {String.fromCharCode(65 + i)}
            </text>
            <text x={PAD / 2} y={x(i) + fontSize / 3}>
              {i + 1}
            </text>
          </g>
        ))}
      </g>

      {hint && (
        <circle
          className="svg-pulse"
          cx={x(hint.col)}
          cy={x(hint.row)}
          r={R * 0.8}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="0.07"
          strokeDasharray="0.2 0.12"
        />
      )}

      {/*
        Stones are fully visible by default; the fade-in is a CSS animation
        *from* a hidden state (index.css, .stone-in), so a browser that does
        not run it still draws every stone.
      */}
      <g>
        {stones.map((s) => (
          <g key={`${s.row}-${s.col}`} className="stone-in">
            {/* Shadow: a plain offset circle, no filter. */}
            <circle cx={x(s.col) + 0.05} cy={x(s.row) + 0.08} r={R} fill="#000" opacity={0.28} />
            <circle
              data-testid="stone"
              cx={x(s.col)}
              cy={x(s.row)}
              r={R}
              fill={`url(#${uid}-${s.stone === BLACK ? "black" : "white"})`}
            />
          </g>
        ))}
      </g>

      {last && !winLine && (
        <circle
          cx={x(last.col)}
          cy={x(last.row)}
          r="0.11"
          fill={grid[last.row]?.[last.col] === BLACK ? "#f8fafc" : "#111318"}
          pointerEvents="none"
        />
      )}

      {winLine && (
        // The winning stones turn blue, one after another along the line,
        // and keep a soft glow; the stones' own colours fade out beneath.
        <g pointerEvents="none">
          {winLine.map((c, i) => (
            <g key={`win-${c.row}-${c.col}`}>
              <circle
                className="fade-in"
                style={{ animationDelay: `${i * 80}ms` }}
                cx={x(c.col)}
                cy={x(c.row)}
                r={R * 1.2}
                fill="var(--win)"
                opacity={0.3}
              />
              <circle
                className="fade-in"
                style={{ animationDelay: `${i * 80}ms` }}
                cx={x(c.col)}
                cy={x(c.row)}
                r={R}
                fill={`url(#${uid}-win)`}
                stroke="#1e3a8a"
                strokeWidth="0.03"
                data-testid="winning-stone"
              />
            </g>
          ))}
        </g>
      )}

      {ghost && turn && (
        <g pointerEvents="none">
          <circle
            cx={x(ghost.col)}
            cy={x(ghost.row)}
            r={R}
            fill={`url(#${uid}-${turn === BLACK ? "black" : "white"})`}
            opacity={0.45}
          />
        </g>
      )}

      {keyboard && (
        <rect
          x={x(cursor.col) - 0.5}
          y={x(cursor.row) - 0.5}
          width="1"
          height="1"
          rx="0.18"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="0.06"
          pointerEvents="none"
        >
          <title>{positionLabel(cursor)}</title>
        </rect>
      )}
    </svg>
  );
}
