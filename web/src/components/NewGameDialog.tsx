import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";
import { PRESETS } from "../game/presets";
import { BLACK, type Player, WHITE } from "../game/rules";
import type { GameSettings } from "../game/state";
import { t } from "../lib/i18n";
import { Button, StoneIcon } from "./ui";

const SIZES = [13, 15, 19] as const;

interface Props {
  open: boolean;
  initial: GameSettings;
  onStart: (settings: GameSettings) => void;
  onClose: () => void;
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

  const option = (active: boolean) =>
    clsx(
      "flex cursor-pointer items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition",
      active ? "border-accent bg-accent/10 ring-1 ring-accent" : "border-line hover:bg-surface",
    );

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
      className="m-auto w-[min(100vw-2rem,28rem)] rounded-3xl border border-line bg-surface-strong p-0 text-fg shadow-2xl"
    >
      <form
        method="dialog"
        className="flex flex-col gap-5 p-5 sm:p-6"
        onSubmit={(e) => {
          e.preventDefault();
          onStart(settings);
        }}
      >
        <h2 className="text-xl font-bold tracking-tight">{t.newGame}</h2>

        <fieldset className="flex flex-col gap-2">
          <legend className="mb-2 text-sm font-semibold text-muted">{t.playAs}</legend>
          <div className="grid grid-cols-2 gap-2">
            {([BLACK, WHITE] as Player[]).map((p) => (
              <label key={p} className={option(settings.human === p)}>
                <input
                  type="radio"
                  name="human"
                  className="sr-only"
                  checked={settings.human === p}
                  onChange={() => setSettings((s) => ({ ...s, human: p }))}
                />
                <StoneIcon stone={p} className="size-6" />
                <span>
                  <span className="block font-semibold">{p === BLACK ? t.black : t.white}</span>
                  {p === BLACK && <span className="block text-xs text-muted">{t.blackFirst}</span>}
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="flex flex-col gap-2">
          <legend className="mb-2 text-sm font-semibold text-muted">{t.difficulty}</legend>
          {PRESETS.map((p) => {
            const [name, desc] = t.presets[p.id];
            return (
              <label key={p.id} className={option(settings.preset === p.id)}>
                <input
                  type="radio"
                  name="preset"
                  className="sr-only"
                  checked={settings.preset === p.id}
                  onChange={() => setSettings((s) => ({ ...s, preset: p.id }))}
                />
                <span className="flex-1">
                  <span className="block font-semibold">{name}</span>
                  <span className="block text-xs text-muted">{desc}</span>
                </span>
                <span className="text-xs tabular-nums text-muted">{t.seconds(p.typicalSeconds)}</span>
              </label>
            );
          })}
        </fieldset>

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
