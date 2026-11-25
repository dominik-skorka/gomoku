import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { configureApi } from "./api/client";
import { config, loadConfig } from "./lib/config";
import { loadLocale } from "./lib/i18n";
import "./index.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root missing from index.html");

// Runtime configuration first (config.json, locales/<lang>.json): both fall
// back to the copies bundled at build time, so the page always renders.
await Promise.all([loadConfig(), loadLocale()]);
configureApi(config.apiBase);

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
