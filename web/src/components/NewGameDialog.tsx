import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";
import { useLevels } from "../game/levels";
import { BLACK, type Player, WHITE } from "../game/rules";
import type { GameSettings, PlayerKind, SideSettings } from "../game/state";
import { t } from "../lib/i18n";
import { Button, StoneIcon } from "./ui";

const SIZES = [13, 15, 19] as const;

interface Props {
  open: boolean;
  initial: GameSettings;
  onStart: (settings: GameSettings) => void;
  onClose: () => void;
}

const option = (active: boolean) =>
  clsx(
    "flex cursor-pointer items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition",
    "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
    active ? "border-accent bg-accent/10 ring-1 ring-accent" : "border-line hover:bg-surface",
  );

/** One colour: human or engine, and the engine's difficulty. */
function SidePicker({
  player,
  value,
  onChange,
}: {
  player: Player;
  value: SideSettings;
  onChange: (v: SideSettings) => void;
}) {
  const name = player === BLACK ? "black" : "white";
  const levels = useLevels();
  return (
    <fieldset className="flex flex-col gap-2 rounded-2xl border border-line p-3">
      <legend className="flex items-center gap-2 px-1 text-sm font-semibold">
        <StoneIcon stone={player} />
        {player === BLACK ? t.black : t.white}
        {player === BLACK && <span className="font-normal text-muted">· {t.blackFirst}</span>}
      </legend>
      <div className="grid grid-cols-2 gap-2">
        {(["human", "engine"] as PlayerKind[]).map((kind) => (
          <label key={kind} className={clsx(option(value.kind === kind), "justify-center py-2")}>
            <input
              type="radio"
              name={`${name}-kind`}
              className="sr-only"
              checked={value.kind === kind}
              onChange={() => onChange({ ...value, kind })}
            />
            <span className="font-semibold">{kind === "human" ? t.human : t.engineKind}</span>
          </label>
        ))}
      </div>
      {value.kind === "engine" && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4" role="radiogroup" aria-label={t.difficulty}>
          {levels.map((p) => {
            const { label, description: desc } = t.level(p.id);
            return (
              <label
                key={p.id}
                className={clsx(option(value.preset === p.id), "flex-col items-start gap-0 py-2")}
                title={desc}
              >
                <input
                  type="radio"
                  name={`${name}-preset`}
                  className="sr-only"
                  checked={value.preset === p.id}
                  onChange={() => onChange({ ...value, preset: p.id })}
                />
                <span className="text-sm font-semibold">{label}</span>
                <span className="text-xs tabular-nums text-muted">{t.seconds(p.typicalSeconds)}</span>
              </label>
            );
          })}
        </div>
      )}
    </fieldset>
  );
}

/** Native <dialog>: focus trap, Esc and backdrop for free, works on every mobile browser. */
export function NewGameDialog({ open, initial, onStart, onClose }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  const [settings, setSettings] = useState(initial);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      setSettings(initial);
      d.showModal();
    } else if (!open && d.open) d.close();
  }, [open, initial]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
      className="m-auto max-h-[calc(100dvh-2rem)] w-[min(100vw-2rem,30rem)] rounded-3xl border border-line bg-surface-strong p-0 text-fg shadow-2xl"
    >
      <form
        method="dialog"
        className="flex flex-col gap-4 p-5 sm:p-6"
        onSubmit={(e) => {
          e.preventDefault();
          onStart(settings);
        }}
      >
        <h2 className="text-xl font-bold tracking-tight">{t.newGame}</h2>

        <div className="flex flex-col gap-3">
          <span className="text-sm font-semibold text-muted">{t.players}</span>
          <SidePicker
            player={BLACK}
            value={settings.black}
            onChange={(black) => setSettings((s) => ({ ...s, black }))}
          />
          <SidePicker
            player={WHITE}
            value={settings.white}
            onChange={(white) => setSettings((s) => ({ ...s, white }))}
          />
        </div>

        <fieldset>
          <legend className="mb-2 text-sm font-semibold text-muted">{t.boardSize}</legend>
          <div className="grid grid-cols-3 gap-2">
            {SIZES.map((n) => (
              <label key={n} className={clsx(option(settings.size === n), "justify-center")}>
                <input
                  type="radio"
                  name="size"
                  className="sr-only"
                  checked={settings.size === n}
                  onChange={() => setSettings((s) => ({ ...s, size: n }))}
                />
                <span className="font-semibold tabular-nums">
                  {n}×{n}
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" onClick={onClose}>
            {t.cancel}
          </Button>
          <Button variant="primary" type="submit" autoFocus>
            {t.start}
          </Button>
        </div>
      </form>
    </dialog>
  );
}
