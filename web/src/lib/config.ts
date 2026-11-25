import bundled from "../../public/config/config.json";

/**
 * Client configuration, owned by the web UI (the API stays generic): where
 * the API is, and the difficulty levels with the engine and parameters each
 * one sends to `POST /v1/move`.
 *
 * It lives in `public/config/config.json`, served at `/config/config.json` and read at
 * start-up, so a deployment can change it (Docker: the file is mounted, see
 * compose.yaml) without rebuilding.  The copy bundled at build time is the
 * fallback when the served file is missing or invalid.
 */
export interface LevelConfig {
  /** Stable key: saved settings, translations (`levels.<id>` in locales) and player bars use it. */
  readonly id: string;
  /** Engine name sent as `brain` (a registry name or a server alias). */
  readonly brain: string;
  /** Sent as `brain_overrides`, merged over the server's settings for that brain. */
  readonly overrides: Readonly<Record<string, unknown>>;
  /** Typical think time shown in the New game dialog. */
  readonly typicalSeconds: number;
}

export interface AppConfig {
  readonly apiBase: string;
  readonly defaultLevel: string;
  readonly levels: readonly LevelConfig[];
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** A valid configuration, or the reasons it is not. */
export function parseConfig(raw: unknown): { config: AppConfig } | { errors: string[] } {
  const errors: string[] = [];
  if (!isRecord(raw)) return { errors: ["config is not a JSON object"] };
  const apiBase = raw.apiBase;
  if (typeof apiBase !== "string" || apiBase === "") errors.push("apiBase must be a non-empty string");
  const levels: LevelConfig[] = [];
  if (!Array.isArray(raw.levels) || raw.levels.length === 0) {
    errors.push("levels must be a non-empty array");
  } else {
    raw.levels.forEach((l: unknown, i) => {
      const where = `levels[${i}]`;
      if (!isRecord(l)) {
        errors.push(`${where} is not an object`);
        return;
      }
      if (typeof l.id !== "string" || !/^[\w-]+$/.test(l.id))
        errors.push(`${where}.id must be letters, digits, _ or -`);
      if (typeof l.brain !== "string" || l.brain === "") errors.push(`${where}.brain must be a non-empty string`);
      if (l.overrides !== undefined && !isRecord(l.overrides)) errors.push(`${where}.overrides must be an object`);
      if (typeof l.typicalSeconds !== "number" || l.typicalSeconds < 0)
        errors.push(`${where}.typicalSeconds must be a number >= 0`);
      if (levels.some((o) => o.id === l.id)) errors.push(`${where}.id "${String(l.id)}" is used twice`);
      levels.push({
        id: String(l.id),
        brain: String(l.brain),
        overrides: isRecord(l.overrides) ? l.overrides : {},
        typicalSeconds: Number(l.typicalSeconds),
      });
    });
  }
  const defaultLevel = raw.defaultLevel;
  if (typeof defaultLevel !== "string" || !levels.some((l) => l.id === defaultLevel))
    errors.push("defaultLevel must be the id of one of the levels");
  if (errors.length > 0) return { errors };
  return { config: { apiBase: apiBase as string, defaultLevel: defaultLevel as string, levels } };
}

const parsedBundle = parseConfig(bundled);
if ("errors" in parsedBundle) throw new Error(`bundled config.json is invalid: ${parsedBundle.errors.join("; ")}`);
const bundledConfig: AppConfig = parsedBundle.config;

/** The configuration in use.  The bundled one until `loadConfig` has run. */
export let config: AppConfig = bundledConfig;

/** Read `/config.json`; if it is missing or invalid, say why and keep the bundled copy. */
export async function loadConfig(fetcher: typeof fetch = fetch): Promise<void> {
  try {
    const response = await fetcher("/config/config.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const parsed = parseConfig(await response.json());
    if ("errors" in parsed) throw new Error(parsed.errors.join("; "));
    config = parsed.config;
  } catch (err) {
    console.warn("gomoku: /config/config.json not used, falling back to the built-in configuration:", err);
    config = bundledConfig;
  }
}
