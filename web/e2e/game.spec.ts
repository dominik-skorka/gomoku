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

test("resign: asks first, then the engine wins and the game is over", async ({ page }) => {
  await stubEngine(page, [
    [0, 0],
    [0, 1],
  ]);
  await startGame(page, "Human", "Engine");
  await tap(page, 7, 7);
  await expect(status(page)).toHaveText(/Your move/);

  const resign = page.getByRole("button", { name: "Resign" }).filter({ visible: true });
  await resign.click();
  const dialog = page.getByRole("dialog", { name: "Resign the game?" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Keep playing" }).click();
  await expect(dialog).toBeHidden();
  await expect(status(page)).toHaveText(/Your move/);

  await resign.click();
  await dialog.getByRole("button", { name: "Resign" }).click();
  await expect(page.getByText("You resigned — the engine wins").filter({ visible: true })).toBeVisible();
  await expect(resign).toBeDisabled();
  await expect(page.getByRole("button", { name: "Undo" }).filter({ visible: true })).toBeDisabled();
});

test("move navigation: view earlier positions without changing the game", async ({ page }) => {
  await stubEngine(page, [[0, 0]]);
  await startGame(page, "Human", "Human");
  await tap(page, 7, 7);
  await tap(page, 6, 6);
  await tap(page, 5, 5);
  const nav = page.getByRole("toolbar", { name: "Move navigation" });
  await expect(nav).toContainText("3 moves");

  await nav.getByRole("button", { name: "Previous move" }).click();
  await expect(nav).toContainText("Viewing 2/3");
  // The board is read-only while viewing.
  await tap(page, 1, 1);
  await expect(nav).toContainText("Viewing 2/3");
  await expect(page.getByRole("heading", { name: /Moves \(3\)/ })).toBeVisible();

  // First move: the position after move 1, not the empty board.
  await nav.getByRole("button", { name: "First move" }).click();
  await expect(nav).toContainText("Viewing 1/3");
  await expect(page.getByRole("button", { name: /^1\.\s*H8/ })).toHaveAttribute("aria-current", "step");
  await expect(nav.getByRole("button", { name: "Previous move" })).toBeDisabled();
  await expect(nav.getByRole("button", { name: "First move" })).toBeDisabled();

  // Clicking a move in the list shows the position after it.
  await page.getByRole("button", { name: /^2\.\s*G7/ }).click();
  await expect(nav).toContainText("Viewing 2/3");
  await page.getByRole("button", { name: /^1\.\s*H8/ }).click();
  await expect(nav).toContainText("Viewing 1/3");
  await expect(page.getByRole("button", { name: /^1\.\s*H8/ })).toHaveAttribute("aria-current", "step");

  // Keyboard: End catches up with the game.
  await page.locator("body").press("End");
  await expect(nav).toContainText("3 moves");
  await expect(status(page)).toHaveText(/White to move/);
});

test("move list follows the game and keeps the move on view visible", async ({ page }) => {
  await stubEngine(page, [[0, 0]]);
  await startGame(page, "Human", "Human");
  // 30 moves, no five in a row: every third column, shifted per row.
  for (let k = 0; k < 30; k++) {
    const row = Math.floor(k / 5);
    await tap(page, row, ((k % 5) * 3 + row) % 15);
  }
  const list = page.getByRole("list").filter({ visible: true }).last();
  const atBottom = () => list.evaluate((el) => Math.abs(el.scrollHeight - el.clientHeight - el.scrollTop) < 2);
  await expect.poll(atBottom).toBe(true);

  // Back to the first move: the list scrolls up to show it.
  await page.getByRole("button", { name: "First move" }).filter({ visible: true }).click();
  const first = page.getByRole("button", { name: /^1\.\s*A1/ });
  await expect(first).toHaveAttribute("aria-current", "step");
  await expect(first).toBeInViewport();
  const visibleInList = await first.evaluate((el) => {
    const box = el.closest("ol")?.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    return !!box && r.top >= box.top && r.bottom <= box.bottom;
  });
  expect(visibleInList).toBe(true);

  // Catch up: back to the bottom.
  await page.getByRole("button", { name: "Latest move" }).filter({ visible: true }).click();
  await expect.poll(atBottom).toBe(true);
});
