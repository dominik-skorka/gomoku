import { clsx } from "clsx";
import { useCallback, useEffect, useMemo, useState } from "react";
import { requestMove } from "./api/client";
import { Board } from "./components/Board";
import { Controls } from "./components/Controls";
import { MoveList } from "./components/MoveList";
import { FIRST, MoveNav } from "./components/MoveNav";
import { NewGameDialog } from "./components/NewGameDialog";
import { PlayerBar } from "./components/PlayerBar";
import { ResignDialog } from "./components/ResignDialog";
import { Button, Card } from "./components/ui";
import { presetById } from "./game/presets";
import { BLACK, type Cell, WHITE } from "./game/rules";
import { canResign, canUndo, type GameSettings, isHumanTurn, migrateSettings, replay, sideToMove } from "./game/state";
import { useGame } from "./game/useGame";
import { t } from "./lib/i18n";
import { statusText } from "./lib/status";
import { useTheme } from "./lib/theme";

const SETTINGS_KEY = "gomoku.settings";

function loadSettings(): GameSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return migrateSettings(raw ? JSON.parse(raw) : null);
  } catch {
    // storage blocked or unreadable: fall back to defaults
    return migrateSettings(null);
  }
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
  const { state, engine, play, newGame, undo, resign, retry, paused, togglePause, step, speed, setSpeed } =
    useGame(initial);
  const [resignOpen, setResignOpen] = useState(false);
  // Position on view: null follows the game; a number shows the board after
  // that many moves without changing the game (D6: view only).
  const [viewPly, setViewPly] = useState<number | null>(null);
  const total = state.moves.length;
  const shownPly = viewPly === null ? total : Math.min(viewPly, total);
  const viewing = shownPly < total;
  const goTo = useCallback((ply: number) => setViewPly(ply >= total ? null : Math.max(0, ply)), [total]);
  const { choice, cycle } = useTheme();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [hint, setHint] = useState<Cell | null>(null);
  const [hintLoading, setHintLoading] = useState(false);

  const humanTurn = isHumanTurn(state);
  const side = sideToMove(state);
  const shownGrid = useMemo(
    () => (viewing ? replay(state.settings, state.moves.slice(0, shownPly)) : state.grid),
    [viewing, state.settings, state.moves, shownPly, state.grid],
  );
  const last = state.moves[shownPly - 1] ?? null;
  const winLine = !viewing && state.outcome?.kind === "win" ? state.outcome.line : null;

  // A new game or an undo past the viewed move returns to the live position.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset on a new game only
  useEffect(() => {
    setViewPly(null);
  }, [state.gameId]);

  // Home / ← / → / End anywhere but on the board (whose arrows move its cursor)
  // or in a dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      const target = e.target as Element | null;
      if (target?.closest("[role=application], dialog, input, textarea, select")) return;
      const to = { Home: FIRST, ArrowLeft: shownPly - 1, ArrowRight: shownPly + 1, End: total }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      goTo(to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shownPly, total, goTo]);

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
        sideToMove: side,
        brain: preset.brain,
        overrides: preset.overrides,
      });
      setHint(move.cell);
    } catch {
      // A failed hint is not worth an error banner.
    } finally {
      setHintLoading(false);
    }
  }, [state.grid, side]);

  const controls = {
    state,
    canUndo: canUndo(state),
    canHint: humanTurn && !hintLoading && !viewing,
    canResign: canResign(state),
    paused,
    speed,
    onNewGame: () => setDialogOpen(true),
    onUndo: undo,
    onHint: askHint,
    onResign: () => setResignOpen(true),
    onTogglePause: togglePause,
    onStep: step,
    onSpeed: setSpeed,
  };
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
          {/* What happens next, for screen readers; the player bars show it visually. */}
          <p className="sr-only" aria-live="polite" data-testid="game-status">
            {statusText(state, engine)}
          </p>
          {/* Chess-style: White's bar above the board, Black's below, each with its clock. */}
          <div className="flex w-full max-w-[min(100%,calc(100dvh-17rem))] flex-col gap-2 lg:max-w-[min(100%,calc(100dvh-19rem))]">
            <PlayerBar player={WHITE} state={state} engine={engine} onRetry={retry} />
            <div className="relative aspect-square w-full rounded-[1.25rem] shadow-[var(--shadow)]">
              <Board
                grid={shownGrid}
                turn={humanTurn && !viewing ? side : null}
                last={last}
                winLine={winLine}
                hint={viewing ? null : hint}
                onPlay={play}
              />
              {!state.started && (
                // Before the first game: nothing moves until New game → Start.
                <div className="absolute inset-0 grid place-items-center rounded-[1.25rem] bg-black/10 backdrop-blur-[1px]">
                  <div className="flex flex-col items-center gap-3 rounded-2xl border border-line bg-surface-strong px-6 py-5 text-center shadow-xl">
                    <span className="text-sm text-muted">{t.pressToStart}</span>
                    <Button variant="primary" className="px-6" onClick={() => setDialogOpen(true)}>
                      {t.newGame}
                    </Button>
                  </div>
                </div>
              )}
            </div>
            <PlayerBar player={BLACK} state={state} engine={engine} onRetry={retry} />
          </div>
          {/* Phone/tablet: navigation right under the board; desktop: above the move list. */}
          <div className="w-full lg:hidden">
            <MoveNav ply={shownPly} total={total} onGo={goTo} />
          </div>
          <div className="w-full lg:hidden">
            <Controls {...controls} />
          </div>
        </section>

        <aside className="flex flex-col gap-4">
          <Card className="hidden flex-col gap-4 lg:flex">
            <Controls {...controls} wide />
          </Card>
          {/* Phones: the card holds only the move list, so hide it while empty. */}
          <Card className={clsx("flex-col gap-3", total > 0 ? "flex" : "hidden lg:flex")}>
            <div className="hidden lg:block">
              <MoveNav ply={shownPly} total={total} onGo={goTo} />
            </div>
            <MoveList moves={state.moves} viewPly={shownPly} onGo={goTo} />
          </Card>
        </aside>
      </main>

      <footer className="mx-auto w-full max-w-6xl px-4 pb-4 text-xs text-muted">
        {t.version} {__APP_VERSION__} ({__GIT_SHA__})
      </footer>

      <ResignDialog
        open={resignOpen}
        onResign={() => {
          resign();
          setResignOpen(false);
        }}
        onClose={() => setResignOpen(false)}
      />

      <NewGameDialog
        open={dialogOpen}
        initial={state.settings}
        onStart={startGame}
        onClose={() => setDialogOpen(false)}
      />
    </div>
  );
}
