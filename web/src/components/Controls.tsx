import { clsx } from "clsx";
import { BLACK, type Player } from "../game/rules";
import { type GameState, modeOf, type SideSettings, soleHuman } from "../game/state";
import { SPEEDS, type Speed } from "../game/useGame";
import { t } from "../lib/i18n";
import { WhiteFlagIcon } from "./icons";
import { Button, StoneIcon } from "./ui";

export interface ControlsProps {
  state: GameState;
  canUndo: boolean;
  canHint: boolean;
  canResign: boolean;
  paused: boolean;
  speed: Speed;
  onNewGame: () => void;
  onUndo: () => void;
  onHint: () => void;
  onResign: () => void;
  onTogglePause: () => void;
  onStep: () => void;
  onSpeed: (s: Speed) => void;
  /** Desktop side panel: "New game" on its own row. */
  wide?: boolean;
}

/**
 * The buttons under (phone) or beside (desktop) the board.  A game with a
 * human has New game / Undo / Hint / Resign; engine vs engine has New game /
 * Pause / Step plus a speed switch.
 */
export function Controls(p: ControlsProps) {
  const selfPlay = modeOf(p.state.settings) === "engine-engine";
  const over = p.state.outcome !== null;
  return (
    <div className="flex w-full flex-col gap-2">
      <div className={clsx("grid gap-2", p.wide || !selfPlay ? "grid-cols-2" : "grid-cols-3")}>
        <Button variant="primary" className={p.wide ? "col-span-2" : undefined} onClick={p.onNewGame}>
          {t.newGame}
        </Button>
        {selfPlay ? (
          <>
            <Button onClick={p.onTogglePause} disabled={over} aria-pressed={p.paused}>
              {p.paused ? t.resume : t.pause}
            </Button>
            <Button onClick={p.onStep} disabled={over || !p.paused}>
              {t.step}
            </Button>
          </>
        ) : (
          <>
            <Button onClick={p.onUndo} disabled={!p.canUndo}>
              {t.undo}
            </Button>
            <Button onClick={p.onHint} disabled={!p.canHint}>
              {t.hint}
            </Button>
            <Button
              className={p.wide ? "col-span-2" : "col-start-2 row-start-1"}
              onClick={p.onResign}
              disabled={!p.canResign}
            >
              <WhiteFlagIcon />
              {t.resign}
            </Button>
          </>
        )}
      </div>
      {selfPlay && (
        <fieldset className="flex items-center gap-2">
          <legend className="sr-only">{t.speed}</legend>
          <span aria-hidden="true" className="text-sm text-muted">
            {t.speed}
          </span>
          <div className="grid flex-1 grid-cols-3 gap-1 rounded-xl border border-line p-1">
            {SPEEDS.map((s, i) => (
              <label
                key={s}
                className={clsx(
                  "cursor-pointer rounded-lg py-1.5 text-center text-xs font-semibold transition",
                  "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent",
                  p.speed === s ? "bg-accent text-accent-fg" : "hover:bg-surface",
                )}
              >
                <input
                  type="radio"
                  name="speed"
                  className="sr-only"
                  checked={p.speed === s}
                  onChange={() => p.onSpeed(s)}
                />
                {t.speedLabels[i]}
              </label>
            ))}
          </div>
        </fieldset>
      )}
    </div>
  );
}

function SideLabel({ player, side, you }: { player: Player; side: SideSettings; you: Player | null }) {
  const name = side.kind === "human" ? (you === player ? t.you : t.human) : t.presets[side.preset][0];
  return (
    <span className="inline-flex items-center gap-1.5">
      <StoneIcon stone={player} className="size-3.5" />
      <span className="font-medium text-fg">{name}</span>
    </span>
  );
}

/** "● You vs ○ Hard" — who plays which colour in this game. */
export function Matchup({ state }: { state: GameState }) {
  const { settings } = state;
  const you = soleHuman(settings);
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted">
      <SideLabel player={BLACK} side={settings.black} you={you} />
      <span>{t.vs}</span>
      <SideLabel player={2} side={settings.white} you={you} />
      <span>
        · {settings.size}×{settings.size}
      </span>
    </div>
  );
}
