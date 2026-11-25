/**
 * Board rules, mirrored from `gomoku.rules` (freestyle: five *or more* in a
 * row wins).  The browser owns the game state, so it must recognise the end
 * of a game itself; the engine is only asked for moves.
 */

export const EMPTY = 0;
export const BLACK = 1;
export const WHITE = 2;

export type Stone = typeof EMPTY | typeof BLACK | typeof WHITE;
export type Player = typeof BLACK | typeof WHITE;
export type Grid = readonly (readonly Stone[])[];

export interface Cell {
  readonly row: number;
  readonly col: number;
}

const DIRECTIONS: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

export function emptyGrid(size: number): Grid {
  return Array.from({ length: size }, () => Array<Stone>(size).fill(EMPTY));
}

export function opponent(player: Player): Player {
  return player === BLACK ? WHITE : BLACK;
}

export function stoneAt(grid: Grid, cell: Cell): Stone {
  return grid[cell.row]?.[cell.col] ?? EMPTY;
}

export function isLegal(grid: Grid, cell: Cell): boolean {
  const size = grid.length;
  return cell.row >= 0 && cell.row < size && cell.col >= 0 && cell.col < size && stoneAt(grid, cell) === EMPTY;
}

/** A new grid with `stone` on `cell`; the input is not modified. */
export function placeStone(grid: Grid, cell: Cell, stone: Player): Grid {
  return grid.map((row, r) => (r === cell.row ? row.map((s, c) => (c === cell.col ? stone : s)) : row));
}

/**
 * The line through `last` of at least `winLength` stones of its colour,
 * ordered from one end to the other, or `null` when the move did not win.
 */
export function winningLine(grid: Grid, last: Cell, winLength = 5): Cell[] | null {
  const stone = stoneAt(grid, last);
  if (stone === EMPTY) return null;
  const size = grid.length;
  for (const [dr, dc] of DIRECTIONS) {
    const line: Cell[] = [last];
    for (const sign of [-1, 1]) {
      let r = last.row + dr * sign;
      let c = last.col + dc * sign;
      while (r >= 0 && r < size && c >= 0 && c < size && grid[r]?.[c] === stone) {
        if (sign < 0) line.unshift({ row: r, col: c });
        else line.push({ row: r, col: c });
        r += dr * sign;
        c += dc * sign;
      }
    }
    if (line.length >= winLength) return line;
  }
  return null;
}

export function isFull(grid: Grid): boolean {
  return grid.every((row) => row.every((s) => s !== EMPTY));
}

/** Engine notation: column letter + 1-based row, e.g. `H8` (matches `Move.position`). */
export function positionLabel(cell: Cell): string {
  return `${String.fromCharCode(65 + cell.col)}${cell.row + 1}`;
}
