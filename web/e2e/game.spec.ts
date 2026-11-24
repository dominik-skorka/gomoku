import { expect, type Page, test } from "@playwright/test";

/** Engine stub: answers each move request with the next cell of `cells`. */
async function stubEngine(page: Page, cells: [number, number][]) {
  const requests: unknown[] = [];
  let i = 0;
  await page.route("**/api/v1/move", async (route) => {
    requests.push(route.request().postDataJSON());
    const [row, col] = cells[i++ % cells.length] as [number, number];
    await route.fulfill({ json: { move: { row, col, position: "" }, brain: "stub", diagnostics: null } });
  });
  return requests;
}

async function startGame(page: Page, black: "Human" | "Engine", white: "Human" | "Engine") {
  await page.goto("/");
  await page.getByRole("button", { name: "New game" }).filter({ visible: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("group", { name: /Black/ }).getByText(black, { exact: true }).click();
  await dialog.getByRole("group", { name: /White/ }).getByText(white, { exact: true }).click();
  await dialog.getByRole("button", { name: "Start" }).click();
  await expect(dialog).toBeHidden();
}

/** Click the intersection at (row, col) of a 15×15 board. */
async function tap(page: Page, row: number, col: number) {
  const board = page.getByRole("application");
  const box = await board.boundingBox();
  if (!box) throw new Error("board not visible");
  const span = 16; // 15 points + 2 × 0.5 margin cells
  const x = box.x + ((1 + col) / span) * box.width;
  const y = box.y + ((1 + row) / span) * box.height;
  // Touch needs a preview tap first, then a confirming tap.
  const isTouch = await page.evaluate(() => matchMedia("(pointer: coarse)").matches);
  if (isTouch) await page.touchscreen.tap(x, y);
  if (isTouch) await page.touchscreen.tap(x, y);
  else await page.mouse.click(x, y);
}

const status = (page: Page) => page.getByText(/Your move|to move|thinking|won|wins|Paused/).filter({ visible: true });

test("human vs engine: five in a row wins and turns blue", async ({ page }) => {
  await stubEngine(page, [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
  ]);
  await startGame(page, "Human", "Engine");
  for (let c = 3; c < 8; c++) {
    await expect(status(page)).toHaveText(/Your move/);
    await tap(page, 7, c);
  }
  await expect(status(page)).toHaveText(/You won!/);
  await expect(page.getByTestId("winning-stone")).toHaveCount(5);
});

test("human vs human: colours alternate on one device and undo takes back one move", async ({ page }) => {
  const requests = await stubEngine(page, [[0, 0]]);
  await startGame(page, "Human", "Human");
  await expect(status(page)).toHaveText(/Black to move/);
  await tap(page, 7, 7);
  await expect(status(page)).toHaveText(/White to move/);
  await tap(page, 6, 6);
  await expect(page.getByRole("heading", { name: /Moves \(2\)/ })).toBeVisible();
  await page.getByRole("button", { name: "Undo" }).filter({ visible: true }).click();
  await expect(page.getByRole("heading", { name: /Moves \(1\)/ })).toBeVisible();
  await expect(status(page)).toHaveText(/White to move/);
  expect(requests).toHaveLength(0); // no engine involved
});

test("engine vs engine: plays itself, pauses, steps one move at a time", async ({ page }) => {
  const cells: [number, number][] = Array.from({ length: 40 }, (_, k) => [Math.floor(k / 8) * 2, k % 8]);
  const requests = await stubEngine(page, cells);
  await startGame(page, "Engine", "Engine");
  await page.getByText("Fast", { exact: true }).filter({ visible: true }).click();
  await expect.poll(() => requests.length).toBeGreaterThan(3);

  await page.getByRole("button", { name: "Pause" }).filter({ visible: true }).click();
  await expect(status(page)).toHaveText(/Paused/);
  const before = requests.length;
  await page.waitForTimeout(500);
  expect(requests.length).toBe(before);

  await page.getByRole("button", { name: "Step" }).filter({ visible: true }).click();
  await expect.poll(() => requests.length).toBe(before + 1);
  await expect(status(page)).toHaveText(/Paused/);
  expect(await page.getByRole("button", { name: "Undo" }).count()).toBe(0);

  await page.getByRole("button", { name: "Resume" }).filter({ visible: true }).click();
  await expect.poll(() => requests.length).toBeGreaterThan(before + 2);
  // Each request is for the side to move, alternating colours.
  const sides = (requests as { side_to_move: number }[]).slice(0, 4).map((r) => r.side_to_move);
  expect(sides).toEqual([1, 2, 1, 2]);
});

test("rate limited: says so, counts down and retries by itself", async ({ page }) => {
  let calls = 0;
  await page.route("**/api/v1/move", async (route) => {
    calls++;
    if (calls === 1) {
      await route.fulfill({
        status: 429,
        headers: { "Retry-After": "2" },
        json: { detail: { error: "rate_limit_exceeded" } },
      });
    } else {
      await route.fulfill({ json: { move: { row: 0, col: 0, position: "A1" }, brain: "stub", diagnostics: null } });
    }
  });
  await startGame(page, "Engine", "Human");
  await expect(page.getByText("Too many requests.").filter({ visible: true })).toBeVisible();
  await expect(page.getByText(/Retrying in \d s/).filter({ visible: true })).toBeVisible();
  await expect(status(page)).toHaveText(/Your move/, { timeout: 5000 });
  expect(calls).toBe(2);
});
