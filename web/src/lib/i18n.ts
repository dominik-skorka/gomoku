import en from "../../public/locales/en.json";

/**
 * UI strings.  They live in `public/locales/<lang>.json`, served next to the
 * page and read at start-up (`loadLocale`), so texts can change without a
 * rebuild.  The language follows the browser (Polish or English).  English,
 * bundled at build time, fills any key a locale file lacks, so a partial
 * translation never shows a blank.
 *
 * `{name}` placeholders become functions: `"{side} to move"` is called as
 * `t.toMove(side)`.
 */
export type Messages = typeof en;

interface LevelText {
  readonly label: string;
  readonly description: string;
}

/** Messages with their templates compiled to functions. */
export interface Strings
  extends Omit<
    Messages,
    | "toMove"
    | "isThinking"
    | "wins"
    | "resigned"
    | "navLive"
    | "navViewing"
    | "retryingIn"
    | "seconds"
    | "secondsUnder1"
    | "levels"
  > {
  toMove(side: string): string;
  isThinking(side: string): string;
  wins(side: string): string;
  resigned(side: string): string;
  navLive(n: number): string;
  navViewing(ply: number, total: number): string;
  retryingIn(s: number): string;
  /** Typical think time: "< 1 s" or "~6 s". */
  seconds(s: number): string;
  /** Label and description of a difficulty level; its id when the locale has none. */
  level(id: string): LevelText;
}

export type Lang = "en" | "pl";

export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) => (key in values ? String(values[key]) : match));
}

/** Deep merge of a (possibly partial) locale over English. */
function withFallback(partial: unknown): Messages {
  const p = (typeof partial === "object" && partial !== null ? partial : {}) as Record<string, unknown>;
  const out: Record<string, unknown> = { ...en };
  for (const [key, value] of Object.entries(p)) {
    const base = (en as Record<string, unknown>)[key];
    if (key === "levels" && typeof value === "object" && value !== null) {
      out.levels = { ...en.levels, ...(value as object) };
    } else if (base !== undefined && typeof value === typeof base && Array.isArray(value) === Array.isArray(base)) {
      out[key] = value;
    }
  }
  return out as Messages;
}

export function compile(messages: Messages): Strings {
  const m = messages;
  const levels = m.levels as Record<string, Partial<LevelText> | undefined>;
  return {
    ...m,
    toMove: (side) => fill(m.toMove, { side }),
    isThinking: (side) => fill(m.isThinking, { side }),
    wins: (side) => fill(m.wins, { side }),
    resigned: (side) => fill(m.resigned, { side }),
    navLive: (n) => fill(m.navLive, { n }),
    navViewing: (ply, total) => fill(m.navViewing, { ply, total }),
    retryingIn: (s) => fill(m.retryingIn, { s }),
    seconds: (s) => (s < 1 ? m.secondsUnder1 : fill(m.seconds, { s })),
    level: (id) => ({ label: levels[id]?.label ?? id, description: levels[id]?.description ?? "" }),
  };
}

export const locale: Lang =
  typeof navigator !== "undefined" && navigator.language.toLowerCase().startsWith("pl") ? "pl" : "en";

/** The current strings.  English until `loadLocale` has run. */
export let t: Strings = compile(en);

/** Read `locales/<lang>.json`; on any failure keep the bundled English. */
export async function loadLocale(lang: Lang = locale, fetcher: typeof fetch = fetch): Promise<void> {
  try {
    const response = await fetcher(`/locales/${lang}.json`, { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    t = compile(withFallback(await response.json()));
  } catch (err) {
    console.warn(`gomoku: locale ${lang} not loaded, using English`, err);
    t = compile(en);
  }
  document.documentElement.lang = lang;
}
