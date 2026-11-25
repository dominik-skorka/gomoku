import { devices, expect, type Page, test } from "@playwright/test";

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

async function startGame(page: Page, black: "Human" | "Computer", white: "Human" | "Computer") {
  await page.goto("/");
  await page.getByRole("button", { name: "New game" }).filter({ visible: true }).first().click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("group", { name: /Black/ }).getByText(black, { exact: true }).click();
  await dialog.getByRole("group", { name: /White/ }).getByText(white, { exact: true }).click();
  await dialog.getByRole("button", { name: "Start" }).click();
  await expect(dialog).toBeHidden();
}

/** Start a game with the saved (default) settings: New game, then Start. */
async function startDefault(page: Page) {
  await page.getByRole("button", { name: "New game" }).filter({ visible: true }).first().click();
  await page.getByRole("dialog").getByRole("button", { name: "Start" }).click();
}

/** Click the intersection at (row, col) of a 15×15 board. */
async function tap(page: Page, row: number, col: number) {
  const board = page.getByRole("application");
  const box = await board.boundingBox();
  if (!box) throw new Error("board not visible");
  const span = 16; // 15 points + 2 × 0.5 margin cells
  const x = box.x + ((1 + col) / span) * box.width;
  const y = box.y + ((1 + row) / span) * box.height;
  const isTouch = await page.evaluate(() => matchMedia("(pointer: coarse)").matches);
  if (isTouch) await page.touchscreen.tap(x, y);
  else await page.mouse.click(x, y);
}

/** The moves in the move list (the visible one on this layout). */
const moveItems = (page: Page) =>
  page.getByRole("list", { name: "Moves" }).filter({ visible: true }).getByRole("listitem");

/** The game status line for screen readers (the player bars show it visually). */
const status = (page: Page) => page.getByTestId("game-status");
const bar = (page: Page, colour: "black" | "white") => page.getByTestId(`player-${colour}`);

test("human vs engine: five in a row wins and turns blue", async ({ page }) => {
  await stubEngine(page, [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
  ]);
  await startGame(page, "Human", "Computer");
  for (let i = 0; i < 5; i++) {
    // Wait for the engine's reply in the move list, not for the status
    // text: while the status animates out, the previous "Your move" is
    // still on screen, and a tap during the engine's turn is ignored.
    await expect(moveItems(page)).toHaveCount(2 * i);
    await tap(page, 7, 3 + i);
  }
  await expect(moveItems(page)).toHaveCount(9);
  await expect(status(page)).toHaveText(/You won!/);
  await expect(bar(page, "black")).toContainText("Wins");
  await expect(bar(page, "white")).not.toContainText("Wins");
  await expect(page.getByTestId("winning-stone")).toHaveCount(5);
});

test("human vs human: colours alternate on one device and undo takes back one move", async ({ page }) => {
  const requests = await stubEngine(page, [[0, 0]]);
  await startGame(page, "Human", "Human");
  await expect(status(page)).toHaveText(/Black to move/);
  await tap(page, 7, 7);
  await expect(status(page)).toHaveText(/White to move/);
  await tap(page, 6, 6);
  await expect(moveItems(page)).toHaveCount(2);
  await page.getByRole("button", { name: "Undo" }).filter({ visible: true }).click();
  await expect(moveItems(page)).toHaveCount(1);
  await expect(status(page)).toHaveText(/White to move/);
  expect(requests).toHaveLength(0); // no engine involved
});

test("engine vs engine: plays itself, pauses, steps one move at a time", async ({ page }) => {
  const cells: [number, number][] = Array.from({ length: 40 }, (_, k) => [Math.floor(k / 8) * 2, k % 8]);
  const requests = await stubEngine(page, cells);
  await startGame(page, "Computer", "Computer");
  await page.getByText("Fast", { exact: true }).filter({ visible: true }).click();
  await expect.poll(() => requests.length).toBeGreaterThan(3);

  await page.getByRole("button", { name: "Pause" }).filter({ visible: true }).click();
  await expect(status(page)).toHaveText(/Paused/);
  await expect(
    page.getByTestId("player-black").or(page.getByTestId("player-white")).filter({ hasText: "Paused" }),
  ).toHaveCount(1);
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
  await startGame(page, "Computer", "Human");
  // The error shows in the bar of the engine whose move failed (Black).
  await expect(bar(page, "black").getByText("Too many requests.")).toBeVisible();
  await expect(bar(page, "black").getByText(/Retrying in \d s/)).toBeVisible();
  await expect(status(page)).toHaveText(/Your move/, { timeout: 5000 });
  expect(calls).toBe(2);
});

test("resign: asks first, then the computer wins and the game is over", async ({ page }) => {
  await stubEngine(page, [
    [0, 0],
    [0, 1],
  ]);
  await startGame(page, "Human", "Computer");
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
  await expect(status(page)).toHaveText("You resigned — the computer wins");
  await expect(bar(page, "white")).toContainText("Wins");
  await expect(bar(page, "black")).toContainText("Resigned");
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
  await expect(nav).toContainText("Moves: 3");

  await nav.getByRole("button", { name: "Previous move" }).click();
  await expect(nav).toContainText("Viewing 2/3");
  // The board is read-only while viewing.
  await tap(page, 1, 1);
  await expect(nav).toContainText("Viewing 2/3");
  await expect(moveItems(page)).toHaveCount(3);

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
  await expect(nav).toContainText("Moves: 3");
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

test("touch: one tap places a stone on tablets and phones", async ({ browser }) => {
  for (const device of ["iPad Pro 11", "iPad Pro 11 landscape", "Pixel 7", "iPhone 15"] as const) {
    const context = await browser.newContext({ ...devices[device], locale: "en-US" });
    const page = await context.newPage();
    await stubEngine(page, [[0, 0]]);
    await page.goto("/");
    await startDefault(page);
    const box = await page.getByRole("application").boundingBox();
    if (!box) throw new Error("board not visible");
    await page.touchscreen.tap(box.x + (8 / 16) * box.width, box.y + (8 / 16) * box.height);
    // The move and the engine's reply, both drawn.
    await expect(moveItems(page)).toHaveCount(2);
    await expect(page.getByTestId("stone")).toHaveCount(2);
    await context.close();
  }
});

test("stones are drawn even when the browser runs no animations", async ({ page }) => {
  await stubEngine(page, [[0, 0]]);
  await page.goto("/");
  await startDefault(page);
  // Simulate a browser that never runs (or never finishes) animations.
  await page.addStyleTag({
    content: "*, *::before, *::after { animation: none !important; transition: none !important; }",
  });
  await tap(page, 7, 7);
  const stones = page.getByTestId("stone");
  await expect(stones).toHaveCount(2); // the move and the engine's reply
  for (const stone of await stones.all()) {
    await expect(stone).toBeVisible();
    expect(await stone.evaluate((el) => getComputedStyle(el).opacity)).toBe("1");
  }
});

test("player bars: White above the board, Black below, clocks run on their own turn", async ({ page }) => {
  const requests = await stubEngine(page, [[0, 0]]);
  await startGame(page, "Human", "Human");
  const board = await page.getByRole("application").boundingBox();
  const white = await bar(page, "white").boundingBox();
  const black = await bar(page, "black").boundingBox();
  if (!board || !white || !black) throw new Error("layout not visible");
  expect(white.y + white.height).toBeLessThanOrEqual(board.y);
  expect(black.y).toBeGreaterThanOrEqual(board.y + board.height);

  // Black to move: Black's clock runs, White's stands at 0:00.
  await expect(bar(page, "black")).toHaveAttribute("data-active", "true");
  await expect(bar(page, "black").getByTestId("clock")).toHaveText("0:01", { timeout: 3000 });
  await expect(bar(page, "white").getByTestId("clock")).toHaveText("0:00");
  await tap(page, 7, 7);
  await expect(bar(page, "white")).toHaveAttribute("data-active", "true");
  const blackAfter = await bar(page, "black").getByTestId("clock").textContent();
  await page.waitForTimeout(1200);
  // Black's clock stopped; White's is running.
  await expect(bar(page, "black").getByTestId("clock")).toHaveText(blackAfter ?? "");
  await expect(bar(page, "white").getByTestId("clock")).not.toHaveText("0:00");
  expect(requests).toHaveLength(0);
});

test("loading or reloading the page does not start a game; New game → Start does", async ({ page }) => {
  const requests = await stubEngine(page, [[7, 7]]);
  // Engine plays Black, so a started game would make it move at once.
  await page.goto("/");
  await page.evaluate(() =>
    localStorage.setItem(
      "gomoku.settings",
      JSON.stringify({
        size: 15,
        winLength: 5,
        black: { kind: "engine", preset: "hard" },
        white: { kind: "human", preset: "hard" },
      }),
    ),
  );
  for (const load of [() => page.reload(), () => page.goto("/")]) {
    await load();
    await expect(status(page)).toHaveText("Press New game to start.");
    await page.waitForTimeout(1200);
    expect(requests).toHaveLength(0);
    await expect(bar(page, "black").getByTestId("clock")).toHaveText("0:00");
    await expect(bar(page, "white").getByTestId("clock")).toHaveText("0:00");
    // The board ignores taps before the game starts.
    await tap(page, 3, 3);
    await expect(moveItems(page)).toHaveCount(0);
  }
  await startDefault(page);
  await expect(moveItems(page)).toHaveCount(1);
  expect(requests).toHaveLength(1);
  await expect(status(page)).toHaveText(/Your move/);
});

const BRAINS = {
  brains: [
    {
      name: "negamax",
      settings_schema: { properties: { depth: { type: "integer", minimum: 1, maximum: 10, default: 3 } } },
    },
    {
      name: "alphabeta",
      settings_schema: {
        properties: {
          depth: { type: "integer", minimum: 1, maximum: 10, default: 4 },
          lmr: { type: "boolean", default: false, description: "Late move reductions." },
          opening_book: { type: "string", default: "" },
        },
      },
    },
    {
      name: "threat",
      settings_schema: {
        properties: {
          pns_time_s: { type: "number", minimum: 0.1, maximum: 60 },
          threats: { $ref: "#/$defs/Search", description: "Which threats." },
        },
        $defs: {
          Search: { properties: { own: { $ref: "#/$defs/Attack" }, defence: { $ref: "#/$defs/Defence" } } },
          Attack: { properties: { fours: { type: "boolean" }, threes: { type: "boolean" } } },
          Defence: { properties: { block_fours: { type: "boolean" } } },
        },
      },
    },
  ],
};

test("config.json decides the levels and what each sends to the engine", async ({ page }) => {
  await page.route("**/config.json", (route) =>
    route.fulfill({
      json: {
        apiBase: "/api",
        defaultLevel: "blitz",
        levels: [{ id: "blitz", brain: "random", overrides: { seed: 1 }, typicalSeconds: 0 }],
      },
    }),
  );
  const requests = await stubEngine(page, [[0, 0]]);
  await page.goto("/");
  await page.getByRole("button", { name: "New game" }).filter({ visible: true }).first().click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("group", { name: /Black/ }).getByText("Computer", { exact: true }).click();
  // A level without a translation shows its id.
  await expect(dialog.getByText("blitz", { exact: true }).first()).toBeVisible();
  await dialog.getByRole("button", { name: "Start" }).click();
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0]).toMatchObject({ brain: "random", brain_overrides: { seed: 1 } });
});

const SERVER_SETTINGS: Record<string, Record<string, unknown>> = {
  negamax: { depth: 3 },
  alphabeta: { depth: 4, lmr: false, opening_book: "" },
  threat: {
    pns_time_s: 5,
    threats: { own: { fours: true, threes: true }, defence: { block_fours: true } },
  },
};

async function stubBrains(page: Page) {
  await page.route("**/api/v1/brains", (route) => route.fulfill({ json: BRAINS }));
  await page.route("**/api/v1/brains/*/settings", (route) => {
    const name = new URL(route.request().url()).pathname.split("/").at(-2) ?? "";
    return route.fulfill({ json: { name, type: name, settings: SERVER_SETTINGS[name] ?? {} } });
  });
}

test("level settings: show the server's values, mark and send changes, restore them", async ({ page }) => {
  await stubBrains(page);
  const requests = await stubEngine(page, [
    [0, 0],
    [0, 1],
    [0, 2],
  ]);
  await page.goto("/");
  await page.getByRole("button", { name: "Difficulty levels" }).click();
  const dialog = page.getByRole("dialog", { name: "Difficulty levels" });
  await dialog.getByRole("tab", { name: /Easy/ }).click();
  await expect(dialog.getByLabel("Algorithm")).toHaveValue("negamax");
  // config.json sets Easy's depth to 2; the server plays 3: marked as changed.
  await expect(dialog.locator("#param-depth")).toHaveValue("2");
  await expect(dialog.getByTestId("param-depth")).toHaveAttribute("data-changed", "true");
  await expect(dialog.getByTestId("param-depth")).toContainText("Default: 3");

  // Another engine: fields show the server's values, nothing marked.
  await dialog.getByLabel("Algorithm").selectOption("alphabeta");
  await expect(dialog.locator("#param-depth")).toHaveValue("4");
  await expect(dialog.getByTestId("param-depth")).not.toHaveAttribute("data-changed");
  await expect(dialog.locator("#param-lmr")).toHaveValue("false");
  // Server file paths are never offered.
  await expect(dialog.locator("#param-opening_book")).toHaveCount(0);
  // Descriptions live in an (i) tooltip: hover with a mouse, tap on touch.
  await expect(dialog.getByText("Late move reductions.")).toHaveCount(0);
  const info = dialog.getByRole("button", { name: "lmr: what is this?" });
  const touch = await page.evaluate(() => matchMedia("(pointer: coarse)").matches);
  if (touch) await info.tap();
  else await info.hover();
  await expect(page.getByRole("tooltip")).toHaveText("Late move reductions.");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("tooltip")).toHaveCount(0);

  await dialog.locator("#param-depth").fill("3");
  await dialog.locator("#param-lmr").selectOption("true");
  await expect(dialog.getByTestId("param-depth")).toHaveAttribute("data-changed", "true");
  await expect(dialog.getByTestId("param-lmr")).toHaveAttribute("data-changed", "true");
  await expect(dialog.getByRole("tab", { name: /Easy/ })).toContainText("edited here");
  // Typing the server's value again is no change; × goes back to the server's value.
  await dialog.locator("#param-depth").fill("4");
  await expect(dialog.getByTestId("param-depth")).not.toHaveAttribute("data-changed");
  await dialog.locator("#param-depth").fill("3");
  await dialog.getByRole("button", { name: "lmr: Restore the default" }).click();
  await expect(dialog.locator("#param-lmr")).toHaveValue("false");
  await expect(dialog.getByTestId("param-lmr")).not.toHaveAttribute("data-changed");
  await dialog.getByRole("button", { name: "Close" }).click();

  // The change survives a reload (saved in this browser) and is what the engine gets.
  await page.reload();
  await page.getByRole("button", { name: "New game" }).filter({ visible: true }).first().click();
  const newGame = page.getByRole("dialog");
  await newGame.getByRole("group", { name: /Black/ }).getByText("Computer", { exact: true }).click();
  await newGame.getByRole("group", { name: /Black/ }).getByText("Easy", { exact: true }).click();
  await newGame.getByRole("group", { name: /White/ }).getByText("Human", { exact: true }).click();
  await newGame.getByRole("button", { name: "Start" }).click();
  await expect.poll(() => requests.length).toBeGreaterThan(0);
  expect(requests.at(-1)).toMatchObject({ brain: "alphabeta", brain_overrides: { depth: 3 } });
  expect((requests.at(-1) as { brain_overrides: object }).brain_overrides).not.toHaveProperty("lmr");

  // Restore all: back to the config's Easy.
  await page.getByRole("button", { name: "Difficulty levels" }).click();
  await dialog.getByRole("button", { name: "Restore all" }).click();
  await dialog.getByRole("tab", { name: /Easy/ }).click();
  await expect(dialog.getByLabel("Algorithm")).toHaveValue("negamax");
  await expect(dialog.locator("#param-depth")).toHaveValue("2");
});

test("level settings: nested groups send the whole group with the change", async ({ page }) => {
  await stubBrains(page);
  const requests = await stubEngine(page, [
    [0, 0],
    [0, 1],
  ]);
  await page.goto("/");
  await page.getByRole("button", { name: "Difficulty levels" }).click();
  const dialog = page.getByRole("dialog", { name: "Difficulty levels" });
  await dialog.getByRole("tab", { name: /Master/ }).click();
  const group = dialog.getByTestId("group-threats");
  await group.locator("summary").click();
  await expect(group.locator("#param-threats-own-fours")).toHaveValue("true");
  await group.locator("#param-threats-own-fours").selectOption("false");
  await expect(group.getByTestId("param-threats-own-fours")).toHaveAttribute("data-changed", "true");
  await dialog.getByRole("button", { name: "Close" }).click();

  await page.getByRole("button", { name: "New game" }).filter({ visible: true }).first().click();
  const newGame = page.getByRole("dialog");
  await newGame.getByRole("group", { name: /Black/ }).getByText("Computer", { exact: true }).click();
  await newGame.getByRole("group", { name: /Black/ }).getByText("Master", { exact: true }).click();
  await newGame.getByRole("group", { name: /White/ }).getByText("Human", { exact: true }).click();
  await newGame.getByRole("button", { name: "Start" }).click();
  await expect.poll(() => requests.length).toBeGreaterThan(0);
  // The API merges overrides one level deep, so the whole group goes: the
  // server's values with the one change.
  expect(requests.at(-1)).toMatchObject({
    brain: "threat",
    brain_overrides: { threats: { own: { fours: false, threes: true }, defence: { block_fours: true } } },
  });

  // Back to the default: the group is no longer overridden at all.
  await page.getByRole("button", { name: "Difficulty levels" }).click();
  await dialog.getByRole("tab", { name: /Master/ }).click();
  // The dialog keeps the section's open/closed state.
  if (!(await group.evaluate((d) => (d as HTMLDetailsElement).open))) await group.locator("summary").click();
  await group.getByRole("button", { name: "threats.own.fours: Restore the default" }).click();
  await expect(group.getByTestId("param-threats-own-fours")).not.toHaveAttribute("data-changed");
  await expect(dialog.getByRole("tab", { name: /Master/ })).not.toContainText("edited here");
});

test("level settings: the window keeps its size between levels, no tab focused on open", async ({ page }) => {
  await stubBrains(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Difficulty levels" }).click();
  const dialog = page.getByRole("dialog", { name: "Difficulty levels" });
  // Opening focuses the dialog, not its first tab (Safari drew a focus ring on "Easy").
  await expect(dialog).toBeFocused();
  await dialog.getByRole("tab", { name: /Master/ }).click();
  await expect(dialog.locator("#param-pns_time_s")).toBeVisible();
  const master = await dialog.boundingBox();
  await dialog.getByRole("tab", { name: /Easy/ }).click();
  await expect(dialog.locator("#param-depth")).toBeVisible();
  const easy = await dialog.boundingBox();
  expect(easy?.height).toBe(master?.height);
  expect(easy?.width).toBe(master?.width);
});
