import { useQuery } from "@tanstack/react-query";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { components } from "../api/schema";
import { exportConfig, type Level, levelChanges, resetAllLevels, setLevelChange, useLevels } from "../game/levels";
import { config } from "../lib/config";
import { t } from "../lib/i18n";
import { InfoTip } from "./InfoTip";
import { Button } from "./ui";

type BrainInfo = components["schemas"]["BrainInfo"];

interface Field {
  /** Path from the top of the settings, e.g. ["threats", "own", "fours"]. */
  readonly path: readonly string[];
  /** The last path element: the name shown. */
  readonly key: string;
  readonly type: "integer" | "number" | "boolean" | "string";
  readonly min?: number;
  readonly max?: number;
  readonly description?: string;
}

/** Settings at one level of nesting: [] for the top level, ["threats", "own"] inside groups. */
export interface Group {
  readonly path: readonly string[];
  readonly description?: string;
  readonly fields: readonly Field[];
}

type Schema = Record<string, unknown>;

/**
 * Settings that name files on the server: a client must not choose them
 * (and the form would not know the server's paths anyway).
 */
const HIDDEN = new Set(["opening_book"]);

/** The object schema behind a `$ref` or a nullable `anyOf: [$ref, null]`, if any. */
function objectSchema(p: Schema, defs: Record<string, Schema>): Schema | null {
  const ref = (s: Schema) => (typeof s.$ref === "string" ? (defs[s.$ref.split("/").at(-1) ?? ""] ?? null) : null);
  if (p.properties) return p;
  const direct = ref(p);
  if (direct) return direct;
  if (Array.isArray(p.anyOf)) {
    for (const variant of p.anyOf as Schema[]) {
      const r = ref(variant) ?? (variant.properties ? variant : null);
      if (r) return r;
    }
  }
  return null;
}

/**
 * The editable settings of a brain, from the JSON Schema GET /v1/brains
 * returns: scalar settings, and nested groups (`$ref`, nullable objects)
 * walked into, top level first.
 */
export function groupsOf(schema: Schema): Group[] {
  const defs = (schema.$defs ?? {}) as Record<string, Schema>;
  const groups: Group[] = [];
  const walk = (node: Schema, path: string[], description: string | undefined, depth: number) => {
    const fields: Field[] = [];
    const nested: [Schema, string, string | undefined][] = [];
    for (const [key, p] of Object.entries((node.properties ?? {}) as Record<string, Schema>)) {
      if (HIDDEN.has(key)) continue;
      const type = p.type;
      const desc = typeof p.description === "string" ? p.description : undefined;
      if (type === "integer" || type === "number" || type === "boolean" || type === "string") {
        fields.push({
          path: [...path, key],
          key,
          type,
          min: typeof p.minimum === "number" ? p.minimum : undefined,
          max: typeof p.maximum === "number" ? p.maximum : undefined,
          description: desc,
        });
      } else {
        const obj = objectSchema(p, defs);
        if (obj && depth < 4) nested.push([obj, key, desc]);
      }
    }
    // A top-level group is listed even without settings of its own, for its description.
    if (fields.length > 0 || path.length === 1) groups.push({ path, description, fields });
    for (const [obj, key, desc] of nested) walk(obj, [...path, key], desc, depth + 1);
  };
  walk(schema, [], undefined, 0);
  return groups;
}

/** Every setting's schema default, nested objects included (a baseline for a group the server leaves unset). */
export function defaultsOf(schema: Schema): Record<string, unknown> {
  const defs = (schema.$defs ?? {}) as Record<string, Schema>;
  const build = (node: Schema, depth: number): Record<string, unknown> => {
    const out: Record<string, unknown> = {};
    for (const [key, p] of Object.entries((node.properties ?? {}) as Record<string, Schema>)) {
      const obj = objectSchema(p, defs);
      if (p.default !== undefined && p.default !== null) out[key] = p.default;
      else if (obj && depth < 4) out[key] = build(obj, depth + 1);
    }
    return out;
  };
  return build(schema, 0);
}

export function getPath(obj: unknown, path: readonly string[]): unknown {
  let cur = obj;
  for (const key of path) {
    if (typeof cur !== "object" || cur === null) return undefined;
    cur = (cur as Record<string, unknown>)[key];
  }
  return cur;
}

/** A copy of `obj` with `value` at `path`, creating objects on the way. */
export function setPath(obj: unknown, path: readonly string[], value: unknown): Record<string, unknown> {
  const root: Record<string, unknown> =
    typeof obj === "object" && obj !== null ? structuredClone(obj as Record<string, unknown>) : {};
  let cur = root;
  path.forEach((key, i) => {
    if (i === path.length - 1) cur[key] = value;
    else {
      const next = cur[key];
      cur[key] = typeof next === "object" && next !== null ? next : {};
      cur = cur[key] as Record<string, unknown>;
    }
  });
  return root;
}

/** Same value, by JSON identity (numbers, booleans, strings). */
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

function format(v: unknown): string {
  if (typeof v === "boolean") return v ? t.yes : t.no;
  if (v === undefined || v === null || v === "") return "—";
  return String(v);
}

/**
 * One setting.  It shows the value a move uses: the level's override, or
 * the server's value (`GET /v1/brains/{name}/settings`).  A value that
 * differs from the server's is marked "changed", with the server's value
 * beside it and a button that goes back to it.
 */
function ParamInput({
  field,
  override,
  server,
  onChange,
}: {
  field: Field;
  /** The level's override, if any. */
  override: unknown;
  /** The server's value; undefined while unknown. */
  server: unknown;
  onChange: (v: unknown | undefined) => void;
}) {
  const id = `param-${field.path.join("-")}`;
  const name = field.path.join(".");
  const effective = override !== undefined ? override : server;
  const changed = override !== undefined && server !== undefined && !same(override, server);
  // Equal to the server's value: no override is needed.
  const commit = (v: unknown | undefined) => onChange(v === undefined || same(v, server) ? undefined : v);

  // Text being typed, committed when it is a valid value (empty: back to the server's).
  const [draft, setDraft] = useState(effective === undefined ? "" : String(effective));
  useEffect(() => setDraft(effective === undefined ? "" : String(effective)), [effective]);

  const box = clsx(
    "h-9 w-full rounded-lg border bg-surface-strong px-2 text-sm",
    changed ? "border-accent ring-1 ring-accent" : "border-line",
  );
  let input: React.ReactNode;
  if (field.type === "boolean") {
    input = (
      <select
        id={id}
        className={box}
        value={effective === undefined ? "" : String(effective)}
        onChange={(e) => commit(e.target.value === "" ? undefined : e.target.value === "true")}
      >
        {effective === undefined && <option value="">{t.serverDefault}</option>}
        <option value="true">{t.yes}</option>
        <option value="false">{t.no}</option>
      </select>
    );
  } else {
    const numeric = field.type !== "string";
    const parse = (text: string): unknown | null => {
      if (!numeric) return text;
      const n = field.type === "integer" ? Number.parseInt(text, 10) : Number(text);
      if (!Number.isFinite(n) || (field.type === "integer" && !/^-?\d+$/.test(text.trim()))) return null;
      if ((field.min !== undefined && n < field.min) || (field.max !== undefined && n > field.max)) return null;
      return n;
    };
    const invalid = draft !== "" && parse(draft) === null;
    input = (
      <input
        id={id}
        type={numeric ? "number" : "text"}
        inputMode={field.type === "integer" ? "numeric" : field.type === "number" ? "decimal" : undefined}
        className={clsx(box, invalid && "border-red-500 ring-1 ring-red-500")}
        placeholder={t.serverDefault}
        min={field.min}
        max={field.max}
        step={field.type === "integer" ? 1 : "any"}
        value={draft}
        aria-invalid={invalid || undefined}
        onChange={(e) => {
          setDraft(e.target.value);
          if (e.target.value === "") return;
          const v = parse(e.target.value);
          if (v !== null) commit(v);
        }}
        onBlur={() => {
          if (draft === "") commit(undefined);
          else if (invalid) setDraft(effective === undefined ? "" : String(effective));
        }}
      />
    );
  }
  const range = field.min !== undefined || field.max !== undefined ? `${field.min ?? ""}–${field.max ?? ""}` : "";
  return (
    <div
      data-testid={id}
      data-changed={changed || undefined}
      className={clsx(
        "grid grid-cols-[minmax(0,1fr)_7rem_1.75rem] items-center gap-x-2 gap-y-0.5 rounded-lg px-1.5 py-1 sm:grid-cols-[minmax(0,1fr)_8.5rem_1.75rem]",
        changed && "bg-accent/10",
      )}
    >
      <div className="min-w-0">
        <span className="flex items-center gap-1.5">
          <label htmlFor={id} className="font-mono text-xs font-semibold break-all">
            {field.key}
          </label>
          {field.description && <InfoTip text={field.description} label={`${name}: ${t.about}`} />}
        </span>
        <span className="block text-xs text-muted">
          {changed ? (
            <span>
              {t.serverValue}: <span className="font-medium tabular-nums">{format(server)}</span>
            </span>
          ) : (
            range && <span className="tabular-nums">{range}</span>
          )}
        </span>
      </div>
      {input}
      {changed ? (
        <button
          type="button"
          className="grid size-7 place-items-center rounded-md text-muted hover:bg-surface hover:text-fg"
          title={`${t.useServerValue}: ${format(server)}`}
          aria-label={`${name}: ${t.useServerValue}`}
          onClick={() => onChange(undefined)}
        >
          ×
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}

function LevelEditor({ level, brains }: { level: Level; brains: readonly BrainInfo[] }) {
  const brain = brains.find((b) => b.name === level.brain);
  const groups = brain ? groupsOf(brain.settings_schema) : [];
  const changed = level.id in levelChanges();
  const names = brains.map((b) => b.name);
  if (!names.includes(level.brain)) names.unshift(level.brain); // a server alias the registry does not list
  // What the server plays with for this brain, as the baseline overrides change.
  const server = useQuery({
    queryKey: ["brain-settings", level.brain],
    staleTime: 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/v1/brains/{name}/settings", {
        params: { path: { name: level.brain } },
      });
      if (!data) throw new Error(JSON.stringify(error));
      return data.settings;
    },
  });
  // A group the server leaves unset (e.g. `fallback: null`) starts from the schema defaults.
  const defaults = brain ? defaultsOf(brain.settings_schema) : {};
  const baseline = (top: string): unknown => server.data?.[top] ?? defaults[top];

  /**
   * A top-level setting is overridden as a whole (the API merges overrides
   * one level deep): a change inside a group sends the group's server
   * values with the change applied, and a group back at the server's
   * values is no longer overridden.
   */
  const update = (path: readonly string[], value: unknown | undefined) => {
    const [top, ...rest] = path;
    if (top === undefined) return;
    const overrides: Record<string, unknown> = { ...level.overrides };
    if (rest.length === 0) {
      if (value === undefined) delete overrides[top];
      else overrides[top] = value;
    } else {
      const leaf = value === undefined ? getPath(baseline(top), rest) : value;
      const group = setPath(overrides[top] ?? baseline(top), rest, leaf);
      if (same(group, server.data?.[top])) delete overrides[top];
      else overrides[top] = group;
    }
    setLevelChange(level.id, { brain: level.brain, overrides });
  };

  const input = (f: Field) => {
    const [top = ""] = f.path;
    const overridden = top in level.overrides;
    return (
      <ParamInput
        key={f.path.join(".")}
        field={f}
        override={overridden ? getPath(level.overrides, f.path) : undefined}
        server={server.data ? getPath({ ...defaults, ...server.data, [top]: baseline(top) }, f.path) : undefined}
        onChange={(v) => update(f.path, v)}
      />
    );
  };
  const top = groups.filter((g) => g.path.length === 0);
  // Nested groups under their top-level setting, e.g. threats › own, threats › defence.
  const sections = new Map<string, Group[]>();
  for (const g of groups.filter((g) => g.path.length > 0)) {
    const key = g.path[0] as string;
    sections.set(key, [...(sections.get(key) ?? []), g]);
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[minmax(0,1fr)_7rem_1.75rem] items-center gap-x-2 px-1.5 sm:grid-cols-[minmax(0,1fr)_8.5rem_1.75rem]">
        <label htmlFor="level-brain" className="text-sm font-semibold">
          {t.engineLabel}
        </label>
        <select
          id="level-brain"
          className="h-9 rounded-lg border border-line bg-surface-strong px-2 text-sm"
          value={level.brain}
          onChange={(e) => setLevelChange(level.id, { brain: e.target.value, overrides: {} })}
        >
          {names.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>
      {server.isError && <p className="px-1.5 text-xs text-muted">{t.serverValuesUnavailable}</p>}
      <div className="flex flex-col gap-1">{top.flatMap((g) => g.fields.map(input))}</div>
      {/* Nested settings need the server's values as their baseline. */}
      {server.data &&
        [...sections].map(([key, subgroups]) => {
          const description = subgroups.find((g) => g.path.length === 1)?.description;
          return (
            <details key={key} className="rounded-xl border border-line" data-testid={`group-${key}`}>
              <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm font-semibold">
                <span className="font-mono">{key}</span>
                {description && <InfoTip text={description} label={`${key}: ${t.about}`} />}
              </summary>
              <div className="flex flex-col gap-2 px-1.5 pb-2">
                {subgroups.map((g) => (
                  <div key={g.path.join(".")} className="flex flex-col gap-1">
                    {g.path.length > 1 && g.fields.length > 0 && (
                      <span className="flex items-center gap-1.5 px-1.5 pt-1 font-mono text-xs text-muted">
                        {g.path.join(" › ")}
                        {g.description && <InfoTip text={g.description} label={`${g.path.join(".")}: ${t.about}`} />}
                      </span>
                    )}
                    {g.fields.map(input)}
                  </div>
                ))}
              </div>
            </details>
          );
        })}
      <div className="px-1.5">
        <Button className="h-8 px-3 text-xs" disabled={!changed} onClick={() => setLevelChange(level.id, null)}>
          {t.restoreLevel}
        </Button>
      </div>
    </div>
  );
}

/**
 * Difficulty levels editor: engine and parameters per level, from the
 * engines' settings schemas (GET /v1/brains).  Changes are this browser's
 * own (levels.ts); "Export" turns them into a config.json for everyone.
 */
export function LevelSettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const levels = useLevels();
  const [selected, setSelected] = useState(config.defaultLevel);
  const brains = useQuery({
    queryKey: ["brains"],
    enabled: open,
    staleTime: Number.POSITIVE_INFINITY,
    queryFn: async () => {
      const { data, error } = await api.GET("/v1/brains");
      if (!data) throw new Error(JSON.stringify(error));
      return data.brains;
    },
  });

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      // Focus the dialog itself, not its first button: Safari otherwise draws
      // its focus ring around the first level tab as if it were selected.
      d.focus();
    } else if (!open && d.open) d.close();
  }, [open]);

  const level = levels.find((l) => l.id === selected) ?? levels[0];
  const changes = levelChanges();

  const download = () => {
    const url = URL.createObjectURL(new Blob([exportConfig()], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
      aria-labelledby="levels-title"
      tabIndex={-1}
      // A fixed height, so switching to a level with few settings does not
      // shrink the window; the settings scroll, the footer stays at the bottom.
      className="m-auto h-[min(calc(100dvh-2rem),46rem)] w-[min(100vw-2rem,34rem)] overflow-y-auto overscroll-contain rounded-3xl border border-line bg-surface-strong p-0 text-fg shadow-2xl outline-none"
    >
      <div className="flex min-h-full flex-col gap-4 p-5 pb-0 sm:p-6 sm:pb-0">
        <h2 id="levels-title" className="text-xl font-bold tracking-tight">
          {t.levelSettings}
        </h2>
        <p className="text-xs text-muted">{t.settingsNote}</p>

        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label={t.levelSettings}>
          {levels.map((l) => (
            <button
              key={l.id}
              type="button"
              role="tab"
              aria-selected={l.id === level?.id}
              onClick={() => setSelected(l.id)}
              className={clsx(
                "rounded-lg border px-3 py-1.5 text-sm font-semibold transition",
                "outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-strong",
                l.id === level?.id ? "border-accent bg-accent/10 ring-1 ring-accent" : "border-line hover:bg-surface",
              )}
            >
              {t.level(l.id).label}
              {l.id in changes && (
                // A dot marks a level edited in this browser; the word is for screen readers.
                <span className="ml-1.5 text-accent" title={t.edited}>
                  <span aria-hidden="true">•</span>
                  <span className="sr-only"> ({t.edited})</span>
                </span>
              )}
            </button>
          ))}
        </div>

        {brains.isPending && <p className="text-sm text-muted">{t.loadingEngines}</p>}
        {brains.isError && (
          <div className="flex items-center gap-3 text-sm">
            <span className="text-red-600 dark:text-red-400">{t.enginesUnavailable}</span>
            <Button className="h-8 px-3" onClick={() => brains.refetch()}>
              {t.retry}
            </Button>
          </div>
        )}
        {brains.data && level && <LevelEditor key={level.id} level={level} brains={brains.data} />}

        {/* Footer stays on screen while the parameter list scrolls. */}
        <div className="sticky bottom-0 -mx-5 mt-auto grid grid-cols-2 gap-2 border-t border-line bg-surface-strong px-5 py-4 sm:-mx-6 sm:flex sm:px-6">
          <Button className="h-9 px-3 text-xs" onClick={resetAllLevels} disabled={Object.keys(changes).length === 0}>
            {t.restoreAll}
          </Button>
          <Button className="h-9 px-3 text-xs" onClick={download} title="config.json">
            {t.exportConfig}
          </Button>
          <Button variant="primary" className="col-span-2 sm:ml-auto" onClick={onClose}>
            {t.close}
          </Button>
        </div>
      </div>
    </dialog>
  );
}
