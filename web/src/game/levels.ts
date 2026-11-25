import { useSyncExternalStore } from "react";
import { config, type LevelConfig } from "../lib/config";

/**
 * Difficulty levels in use: the configuration's levels (`config.json`) with
 * this browser's changes from the Level settings dialog on top.  The
 * changes are stored per browser (localStorage) and replace a level's
 * engine and parameters; "Restore defaults" drops them.
 */
export type Level = LevelConfig;

export interface LevelChange {
  readonly brain: string;
  readonly overrides: Readonly<Record<string, unknown>>;
}

const KEY = "gomoku.levels";
type Changes = Readonly<Record<string, LevelChange>>;

function readChanges(): Changes {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "{}") as unknown;
    if (typeof raw !== "object" || raw === null) return {};
    const out: Record<string, LevelChange> = {};
    for (const [id, c] of Object.entries(raw as Record<string, unknown>)) {
      const change = c as Partial<LevelChange> | null;
      if (change && typeof change.brain === "string" && typeof change.overrides === "object" && change.overrides) {
        out[id] = { brain: change.brain, overrides: change.overrides };
      }
    }
    return out;
  } catch {
    return {};
  }
}

let changes: Changes = readChanges();
const listeners = new Set<() => void>();

function save(next: Changes) {
  changes = next;
  try {
    if (Object.keys(next).length === 0) localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Storage blocked: the change lasts for this page only.
  }
  for (const l of listeners) l();
}

export function levelChanges(): Changes {
  return changes;
}

export function setLevelChange(id: string, change: LevelChange | null): void {
  const next = { ...changes };
  if (change === null) delete next[id];
  else next[id] = change;
  save(next);
}

export function resetAllLevels(): void {
  save({});
}

/** The levels in use, in configuration order. */
export function levels(): readonly Level[] {
  return config.levels.map((l) => {
    const c = changes[l.id];
    return c ? { ...l, brain: c.brain, overrides: c.overrides } : l;
  });
}

/** A level by id; the default level for an id the configuration no longer has. */
export function levelById(id: string): Level {
  const all = levels();
  return all.find((l) => l.id === id) ?? all.find((l) => l.id === config.defaultLevel) ?? (all[0] as Level);
}

/** Re-renders when this browser's level changes do. */
export function useLevels(): readonly Level[] {
  useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => changes,
  );
  return levels();
}

/** The levels in use as a `config.json`, to make them the default for everyone. */
export function exportConfig(): string {
  return `${JSON.stringify({ ...config, levels: levels() }, null, 2)}\n`;
}
