import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests of the UI against a stubbed API (each test answers
 * `/api/v1/move` itself), so they run without the Python server and are
 * deterministic.  The real API is covered by its own test suite.
 *
 *   pnpm test:e2e
 *
 * PW_CHROMIUM_PATH points at an already installed Chromium when its version
 * differs from this Playwright's (otherwise: `pnpm exec playwright install chromium`).
 */
const port = 5199;
const executablePath = process.env.PW_CHROMIUM_PATH || undefined;

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: `http://localhost:${port}`,
    locale: "en-US",
    trace: "off",
    launchOptions: { executablePath },
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], launchOptions: { executablePath } } },
    { name: "phone", use: { ...devices["Pixel 7"], launchOptions: { executablePath } } },
  ],
  webServer: {
    command: `pnpm exec vite --port ${port} --strictPort`,
    port,
    reuseExistingServer: !process.env.CI,
  },
});
