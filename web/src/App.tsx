import { useCallback, useEffect, useState } from "react";
import { requestMove } from "./api/client";
import { Board } from "./components/Board";
import { GameStatus } from "./components/GameStatus";
import { MoveList } from "./components/MoveList";
import { NewGameDialog } from "./components/NewGameDialog";
import { Button, Card } from "./components/ui";
import { presetById } from "./game/presets";
import type { Cell } from "./game/rules";
import { canUndo, DEFAULT_SETTINGS, type GameSettings, sideToMove } from "./game/state";
import { useGame } from "./game/useGame";
import { t } from "./lib/i18n";
import { useTheme } from "./lib/theme";

const SETTINGS_KEY = "gomoku.settings";

function loadSettings(): GameSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<GameSettings>) };
  } catch {
    // ignore: fall back to defaults
  }
  return DEFAULT_SETTINGS;
}

function ThemeIcon({ choice }: { choice: "system" | "light" | "dark" }) {
  const common = {
    className: "size-5",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  if (choice === "light")
    return (
      <svg viewBox="0 0 24 24" {...common} aria-hidden="true">
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
    );
  if (choice === "dark")
    return (
      <svg viewBox="0 0 24 24" {...common} aria-hidden="true">
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
      </svg>
    );
  return (
    <svg viewBox="0 0 24 24" {...common} aria-hidden="true">
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M8 20h8M12 16v4" />
    </svg>
  );
}

export function App() {
  const [initial] = useState(loadSettings);
  const { state, engine, play, newGame, undo, retry } = useGame(initial);
  const { choice, cycle } = useTheme();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [hint, setHint] = useState<Cell | null>(null);
  const [hintLoading, setHintLoading] = useState(false);

  const humanTurn = state.outcome === null && sideToMove(state) === state.settings.human;
  const last = state.moves.at(-1) ?? null;
  const winLine = state.outcome?.kind === "win" ? state.outcome.line : null;

  // A hint is only valid for the position it was asked for.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset whenever the position changes
  useEffect(() => {
    setHint(null);
  }, [state.moves.length, state.gameId]);

  const startGame = useCallback(
    (settings: GameSettings) => {
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
      } catch {
        // storage blocked: settings last for this page only
      }
      newGame(settings);
      setDialogOpen(false);
    },
    [newGame],
  );

  const askHint = useCallback(async () => {
    setHintLoading(true);
    try {
      const preset = presetById("hard");
      const move = await requestMove({
        grid: state.grid,
        sideToMove: state.settings.human,
        brain: preset.brain,
        overrides: preset.overrides,
      });
      setHint(move.cell);
    } catch {
      // A failed hint is not worth an error banner.
    } finally {
      setHintLoading(false);
    }
  }, [state.grid, state.settings.human]);

  const [presetName] = t.presets[state.settings.preset];
  const themeLabel = { system: t.themeSystem, light: t.themeLight, dark: t.themeDark }[choice];

  return (
    <div className="flex min-h-dvh flex-col pb-[env(safe-area-inset-bottom)]">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 pt-4 sm:pt-6">
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-extrabold tracking-tight">Gomoku</h1>
          <span className="hidden text-sm text-muted sm:inline">{t.tagline}</span>
        </div>
        <Button
          variant="ghost"
          icon
          onClick={cycle}
          title={`${t.theme}: ${themeLabel}`}
          aria-label={`${t.theme}: ${themeLabel}`}
        >
          <ThemeIcon choice={choice} />
        </Button>
      </header>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-4 px-4 py-4 sm:gap-6 sm:py-6 lg:grid-cols-[minmax(0,1fr)_20rem] lg:items-start">
        <section className="flex flex-col items-center gap-3">
          <div className="w-full lg:hidden">
            <GameStatus state={state} engine={engine} onRetry={retry} />
          </div>
          <div className="aspect-square w-full max-w-[min(100%,calc(100dvh-13rem))] rounded-[1.25rem] shadow-[var(--shadow)] lg:max-w-[min(100%,calc(100dvh-8rem))]">
            <Board
              grid={state.grid}
              turn={humanTurn ? state.settings.human : null}
              last={last}
              winLine={winLine}
              hint={hint}
              onPlay={play}
            />
          </div>
          <div className="grid w-full grid-cols-3 gap-2 lg:hidden">
            <Button variant="primary" onClick={() => setDialogOpen(true)}>
              {t.newGame}
            </Button>
            <Button onClick={undo} disabled={!canUndo(state)}>
              {t.undo}
            </Button>
            <Button onClick={askHint} disabled={!humanTurn || hintLoading}>
              {t.hint}
            </Button>
          </div>
        </section>

        <aside className="flex flex-col gap-4">
          <Card className="hidden flex-col gap-4 lg:flex">
            <GameStatus state={state} engine={engine} onRetry={retry} />
            <div className="text-sm text-muted">
              {t.difficulty}: <span className="font-medium text-fg">{presetName}</span> · {state.settings.size}×
              {state.settings.size}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button variant="primary" className="col-span-2" onClick={() => setDialogOpen(true)}>
                {t.newGame}
              </Button>
              <Button onClick={undo} disabled={!canUndo(state)}>
                {t.undo}
              </Button>
              <Button onClick={askHint} disabled={!humanTurn || hintLoading}>
                {t.hint}
              </Button>
            </div>
          </Card>
          <Card>
            <h2 className="mb-3 text-sm font-semibold text-muted">
              {t.moves} <span className="tabular-nums">({state.moves.length})</span>
            </h2>
            <MoveList moves={state.moves} />
          </Card>
        </aside>
      </main>

      <footer className="mx-auto w-full max-w-6xl px-4 pb-4 text-xs text-muted">
        {t.version} {__APP_VERSION__} ({__GIT_SHA__})
      </footer>

      <NewGameDialog
        open={dialogOpen}
        initial={state.settings}
        onStart={startGame}
        onClose={() => setDialogOpen(false)}
      />
    </div>
  );
}
