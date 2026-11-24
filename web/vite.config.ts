/// <reference types="vitest/config" />
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8")) as { version: string };

/** Git revision baked into the bundle; the Docker build passes it as an env var. */
function revision(): string {
  if (process.env.GIT_SHA) return process.env.GIT_SHA;
  try {
    return execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
  } catch {
    return "unknown";
  }
}

// The dev server proxies /api to a locally running `uv run gomoku-api`, so the
// browser talks to one origin in development exactly as it does behind Caddy.
const apiTarget = process.env.API_TARGET ?? "http://127.0.0.1:8000";

const version = process.env.APP_VERSION ?? pkg.version;
const sha = revision().slice(0, 12);

/** Emits /version.json so monitoring can tell which build is live without parsing JS. */
const versionFile: Plugin = {
  name: "gomoku-version-file",
  generateBundle() {
    this.emitFile({
      type: "asset",
      fileName: "version.json",
      source: `${JSON.stringify({ component: "web", version, revision: sha, built: new Date().toISOString() }, null, 2)}\n`,
    });
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss(), versionFile],
  define: {
    __APP_VERSION__: JSON.stringify(version),
    __GIT_SHA__: JSON.stringify(sha),
  },
  server: {
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
  preview: {
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
  build: { target: "es2023", sourcemap: true },
  test: { environment: "jsdom", include: ["src/**/*.test.{ts,tsx}"] },
});
