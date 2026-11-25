import { describe, expect, it } from "vitest";
import { BLACK, emptyGrid, type Grid, isFull, isLegal, placeStone, positionLabel, WHITE, winningLine } from "./rules";

function play(grid: Grid, cells: [number, number][], stone: typeof BLACK | typeof WHITE): Grid {
  return cells.reduce((g, [row, col]) => placeStone(g, { row, col }, stone), grid);
}

describe("rules", () => {
  it("places stones without mutating the input", () => {
    const g = emptyGrid(15);
    const g2 = placeStone(g, { row: 7, col: 7 }, BLACK);
    expect(g[7]?.[7]).toBe(0);
    expect(g2[7]?.[7]).toBe(BLACK);
    expect(isLegal(g2, { row: 7, col: 7 })).toBe(false);
    expect(isLegal(g2, { row: 15, col: 0 })).toBe(false);
  });

  it.each([
    ["horizontal", [0, 1]],
    ["vertical", [1, 0]],
    ["diagonal", [1, 1]],
    ["anti-diagonal", [1, -1]],
  ] as const)("detects a %s five from its middle stone", (_, [dr, dc]) => {
    const cells = [0, 1, 2, 3, 4].map((i) => [5 + dr * i, 7 + dc * i] as [number, number]);
    const g = play(emptyGrid(15), cells, WHITE);
    const line = winningLine(g, { row: 5 + dr * 2, col: 7 + dc * 2 });
    expect(line).toHaveLength(5);
    expect(line?.[0]).toEqual({ row: 5, col: 7 });
  });

  it("counts an overline as a win (freestyle) and four as no win", () => {
    const six = play(
      emptyGrid(15),
      [0, 1, 2, 3, 4, 5].map((c) => [0, c] as [number, number]),
      BLACK,
    );
    expect(winningLine(six, { row: 0, col: 5 })).toHaveLength(6);
    const four = play(
      emptyGrid(15),
      [0, 1, 2, 3].map((c) => [0, c] as [number, number]),
      BLACK,
    );
    expect(winningLine(four, { row: 0, col: 3 })).toBeNull();
  });

  it("does not join stones of different colours", () => {
    let g = play(
      emptyGrid(9),
      [
        [4, 0],
        [4, 1],
        [4, 3],
        [4, 4],
      ],
      BLACK,
    );
    g = placeStone(g, { row: 4, col: 2 }, WHITE);
    expect(winningLine(g, { row: 4, col: 2 })).toBeNull();
  });

  it("knows a full board and engine notation", () => {
    const full = emptyGrid(5).map((row, r) => row.map((_, c) => ((r + c) % 2 ? BLACK : WHITE)));
    expect(isFull(full)).toBe(true);
    expect(positionLabel({ row: 7, col: 7 })).toBe("H8");
    expect(positionLabel({ row: 4, col: 4 })).toBe("E5");
  });
});
