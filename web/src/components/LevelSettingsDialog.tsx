import { useQuery } from "@tanstack/react-query";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { components } from "../api/schema";
import { exportConfig, type Level, levelChanges, resetAllLevels, setLevelChange, useLevels } from "../game/levels";
import { config } from "../lib/config";
import { t } from "../lib/i18n";
import { Button } from "./ui";

type BrainInfo = components["schemas"]["BrainInfo"];

interface Field {
  readonly key: string;
  readonly type: "integer" | "number" | "boolean" | "string";
  readonly min?: number;
  readonly max?: number;
  readonly description?: string;
}

/**
 * Settings that name files on the server: a client must not choose them
 * (and the form would not know the server's paths anyway).
 */
const HIDDEN = new Set(["opening_book"]);

/** The flat, editable settings of a brain, from the JSON Schema GET /v1/brains returns. */
export function fieldsOf(schema: Record<string, unknown>): { fields: Field[]; hasNested: boolean } {
  const props = (schema.properties ?? {}) as Record<string, Record<string, unknown>>;
  const fields: Field[] = [];
  let hasNested = false;
  for (const [key, p] of Object.entries(props)) {
    if (HIDDEN.has(key)) continue;
    const type = p.type;
    if (type === "integer" || type === "number" || type === "boolean" || type === "string") {
      fields.push({
        key,
        type,
        min: typeof p.minimum === "number" ? p.minimum : undefined,
        max: typeof p.maximum === "number" ? p.maximum : undefined,
        description: typeof p.description === "string" ? p.description : undefined,
      });
    } else {
      hasNested = true;
    }
  }
  return { fields, hasNested };
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
  const id = `param-${field.key}`;
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
        "grid grid-cols-[minmax(0,1fr)_8.5rem_1.75rem] items-center gap-x-2 gap-y-0.5 rounded-lg px-1.5 py-1",
        changed && "bg-accent/10",
      )}
    >
      <label htmlFor={id} className="min-w-0">
        <span className="flex items-center gap-1.5">
          <span className="truncate font-mono text-xs font-semibold">{field.key}</span>
          {changed && <span className="shrink-0 text-[11px] font-semibold text-accent">{t.changed}</span>}
        </span>
        <span className="block truncate text-xs text-muted" title={field.description}>
          {changed ? (
            <span>
              {t.serverValue}: <span className="font-medium tabular-nums">{format(server)}</span>
            </span>
          ) : (
            <>
              {range && <span className="tabular-nums">{range}</span>}
              {range && field.description && " · "}
              {field.description}
            </>
          )}
        </span>
      </label>
      {input}
      {changed ? (
        <button
          type="button"
          className="grid size-7 place-items-center rounded-md text-muted hover:bg-surface hover:text-fg"
          title={`${t.useServerValue}: ${format(server)}`}
          aria-label={`${field.key}: ${t.useServerValue}`}
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
  const { fields, hasNested } = brain ? fieldsOf(brain.settings_schema) : { fields: [], hasNested: false };
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

  const update = (key: string, value: unknown | undefined) => {
    const overrides: Record<string, unknown> = { ...level.overrides };
    if (value === undefined) delete overrides[key];
    else overrides[key] = value;
    setLevelChange(level.id, { brain: level.brain, overrides });
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[minmax(0,1fr)_8.5rem_1.75rem] items-center gap-x-2 px-1.5">
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
      <div className="flex flex-col gap-1">
        {fields.map((f) => (
          <ParamInput
            key={f.key}
            field={f}
            override={level.overrides[f.key]}
            server={server.data?.[f.key]}
            onChange={(v) => update(f.key, v)}
          />
        ))}
        {hasNested && <p className="px-1.5 text-xs text-muted">{t.nestedNote}</p>}
      </div>
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
    if (open && !d.open) d.showModal();
    else if (!open && d.open) d.close();
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
      className="m-auto max-h-[calc(100dvh-2rem)] w-[min(100vw-2rem,34rem)] overflow-y-auto overscroll-contain rounded-3xl border border-line bg-surface-strong p-0 text-fg shadow-2xl"
    >
      <div className="flex flex-col gap-4 p-5 pb-0 sm:p-6 sm:pb-0">
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
                l.id === level?.id ? "border-accent bg-accent/10 ring-1 ring-accent" : "border-line hover:bg-surface",
              )}
            >
              {t.level(l.id).label}
              {l.id in changes && <span className="ml-1.5 text-xs font-normal text-accent">• {t.edited}</span>}
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
        <div className="sticky bottom-0 -mx-5 grid grid-cols-2 gap-2 border-t border-line bg-surface-strong px-5 py-4 sm:-mx-6 sm:flex sm:px-6">
          <Button className="h-9 px-3 text-xs" onClick={resetAllLevels} disabled={Object.keys(changes).length === 0}>
            {t.restoreAll}
          </Button>
          <Button className="h-9 px-3 text-xs" onClick={download} title="config.json">
            {t.exportConfig}
          </Button>
          <Button variant="primary" className="col-span-2 sm:ml-auto" onClick={onClose} autoFocus>
            {t.close}
          </Button>
        </div>
      </div>
    </dialog>
  );
}
