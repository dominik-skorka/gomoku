"""Gomoku settings — pydantic-settings based configuration.

Settings are loaded with the following priority (highest first):

1. Explicit ``__init__`` kwargs (programmatic overrides / CLI)
2. Environment variables with ``GOMOKU_`` prefix
3. ``gomoku.toml`` file discovered via :func:`find_config_file` —
   ``GOMOKU_CONFIG`` env var if set, otherwise a walk-up from the
   current working directory toward the filesystem root
4. Field defaults

Per-brain settings models are defined here, but the mapping from brain
name to settings model lives alongside the brain classes themselves in
``gomoku_brain.BRAINS`` (see :mod:`gomoku_brain` for the single registry).
"""

import os
from pathlib import Path
from typing import override

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

CONFIG_FILENAME = "gomoku.toml"
"""Canonical config filename — searched for during the walk-up."""

# Markers that signal a project root.  When the walk-up encounters one of
# these in a directory that does *not* also contain ``gomoku.toml``, the
# search stops — climbing further up would leak into an unrelated parent
# project.
_PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_config_file(start: Path | None = None) -> Path | None:
    """Locate the active ``gomoku.toml`` for the current process.

    Resolution order:

    1. ``GOMOKU_CONFIG`` environment variable, if set and pointing at an
       existing file.  When set but the file is missing, returns
       ``None`` *without* falling through to the walk-up — the user
       explicitly opted in to that path so silently picking a different
       config would be surprising.
    2. Walk up from *start* (default: current working directory) toward
       the filesystem root.  At each directory:

       * if it contains ``gomoku.toml``, return that path;
       * else if it contains ``pyproject.toml`` or ``.git``, stop —
         we're at the project root and there is no config here.

    Returns ``None`` when no config exists anywhere in the search path.
    Callers (notably :class:`GomokuSettings`) should treat ``None`` as
    "use field defaults".
    """
    env_path = os.environ.get("GOMOKU_CONFIG", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.is_file() else None

    cwd = (start or Path.cwd()).resolve()
    for directory in (cwd, *cwd.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        for marker in _PROJECT_ROOT_MARKERS:
            if (directory / marker).exists():
                # Hit a project boundary without finding the config —
                # don't leak into unrelated parent projects.
                return None
    return None


# ── Per-brain configuration sections ────────────────────────────────────────


_PROBE_THREADS_DOC = (
    "Threads that check candidate moves in parallel for the opponent's forced win (breaker "
    "probes).  The move picked is the one a single thread would pick from the same results, "
    "but more threads probe more candidates within the time cap.  Only a free-threaded "
    "Python (python3.14t) runs them truly in parallel; with the GIL, keep 1."
)


class BaseBrainSettings(BaseModel):
    """Base of every brain's settings model (and of its nested tables).

    Unknown keys are errors: a misspelt setting in ``gomoku.toml``, on the
    command line or in code stops with a message naming it, instead of
    being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class NoBrainSettings(BaseBrainSettings):
    """Settings of a brain that has none."""


class KeyboardBrainSettings(BaseBrainSettings):
    """Settings for the keyboard (human) provider (no tunables)."""


class RandomBrainSettings(BaseBrainSettings):
    """Settings for the random brain (no tunables)."""


class MinimaxBrainSettings(BaseBrainSettings):
    """Settings for the minimax brain."""

    depth: int = Field(default=3, ge=1, le=10)


class NegamaxBrainSettings(BaseBrainSettings):
    """Settings for the negamax brain."""

    depth: int = Field(default=3, ge=1, le=10)


class AlphaBetaBrainSettings(BaseBrainSettings):
    """Settings for the alpha-beta brain."""

    depth: int = Field(default=4, ge=1, le=10)
    vcf_depth: int = Field(default=0, ge=0, le=40)
    qs_vcf_depth: int = Field(default=0, ge=0, le=40)
    threat_ordering: bool = Field(default=True)
    threat_extensions: int = Field(default=0, ge=0, le=8)
    null_move_pruning: bool = Field(default=False)
    null_move_reduction: int = Field(default=2, ge=1, le=4)
    lmr: bool = Field(default=False)
    lmr_threshold: int = Field(default=3, ge=2, le=10)
    lmr_min_depth: int = Field(default=3, ge=2, le=10)
    early_game_radius: int = Field(default=1, ge=1, le=2)
    early_game_threshold: int = Field(default=6, ge=2, le=15)
    early_game_max_candidates: int = Field(default=10, ge=5, le=20)
    early_game_lmr_threshold: int = Field(default=2, ge=1, le=10)
    early_game_futility_margin: int = Field(default=800, ge=200, le=1500)
    max_candidates: int = Field(default=20, ge=5, le=60, description="Branching cap after the early game.")
    futility_margin: int = Field(default=1500, ge=200, le=5000, description="Futility margin after the early game.")
    eval_ordering_depth: int = Field(
        default=3,
        ge=1,
        le=11,
        description="Nodes at least this deep order moves by full evaluation; shallower ones by pattern priors.",
    )
    time_limit_s: float = Field(default=0.0, ge=0.0, le=600.0)
    opening_book: str = Field(default="", description="Path to an opening book JSON file.")
    vcf_breaker_pns_nodes: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        description=(
            "When the opponent has a VCF, PNS budget per breaker for checking it leaves no other "
            "forced win (breakers tried best evaluation first).  0 picks by evaluation only."
        ),
    )
    vcf_breaker_time_s: float = Field(
        default=2.0, ge=0.1, le=60.0, description="Time cap for all the breaker probes of one move."
    )
    probe_threads: int = Field(default=1, ge=1, le=64, description=_PROBE_THREADS_DOC)


class ThreatAttackSettings(BaseBrainSettings):
    """Threats an attacker may force with in VCF / PNS / VCT.

    Fewer only make the search less complete (fewer wins found, never false
    ones); more find more wins but widen the search.
    """

    fours: bool = Field(default=True, description="Fours. Off: no VCF, PNS / VCT attack with threes only.")
    split_fours: bool = Field(default=True, description="Split fours (XXX_X, XX_XX) count as fours.")
    threes: bool = Field(default=True, description="Threes. Off: PNS / VCT attack with fours only.")
    split_threes: bool = Field(
        default=False, description="Split threes (_XX_X_) count as threes; widens the search a lot."
    )


class ThreatDefenceSettings(BaseBrainSettings):
    """Replies the defender gets inside the searches.

    Turning one off makes proofs unsound: the search can prove wins that
    kind of reply refutes, and the brain plays them.
    """

    block_fours: bool = Field(default=True, description="Block the attacker's four. Off: every four 'wins'.")
    block_threes: bool = Field(default=True, description="Block the attacker's three (ends; gap and ends).")
    counter_fours: bool = Field(default=True, description="The defender's own fours as replies.")
    split_counter_fours: bool = Field(default=True, description="Split fours count as counter-fours.")
    counter_threes: bool = Field(default=True, description="The defender's own threes as replies (PNS).")
    split_counter_threes: bool = Field(default=False, description="Split threes count as counter-threes.")
    unblockable_four_defences: bool = Field(
        default=True, description="Cells that defuse an attacker's threat of an unblockable four (PNS)."
    )


class ThreatForkSettings(BaseBrainSettings):
    """How the threat brain finds the opponent's fork cells to block."""

    split: bool = Field(default=False, description="Count split threes and fours (XX_X, XXX_X) as fork lines.")


class ThreatSearchSettings(BaseBrainSettings):
    """[brains.<alias>.threats]: the threat spaces of the threat brain's searches."""

    own: ThreatAttackSettings = Field(default_factory=ThreatAttackSettings)
    opponent: ThreatAttackSettings = Field(default_factory=ThreatAttackSettings)
    defence: ThreatDefenceSettings = Field(default_factory=ThreatDefenceSettings)
    forks: ThreatForkSettings = Field(default_factory=ThreatForkSettings)


class ThreatBrainSettings(BaseBrainSettings):
    """Settings for the threat brain."""

    depth: int = Field(default=4, ge=1, le=10)
    vcf_depth: int = Field(default=20, ge=1, le=40)
    vct_depth: int = Field(default=12, ge=1, le=30)
    threat_search: str = Field(default="pns", pattern="^(pns|vct)$")
    pns_nodes: int = Field(default=10_000, ge=1_000, le=10_000_000)
    pns_time_s: float = Field(default=5.0, ge=0.1, le=60.0)
    probe_threads: int = Field(default=1, ge=1, le=64, description=_PROBE_THREADS_DOC)
    threats: ThreatSearchSettings = Field(
        default_factory=ThreatSearchSettings,
        description=(
            "Which threats the forced-win searches play and answer ([brains.<alias>.threats.*] "
            "tables): own / opponent attack threats, the defender's replies, fork cells."
        ),
    )
    fallback: AlphaBetaBrainSettings | None = Field(
        default=None,
        description=(
            "Tuning for the alpha-beta fallback used when no forced win is found "
            "([brains.<alias>.fallback] table). None means a default-configured "
            "fallback at the top-level depth; when set, its own depth field "
            "(default 4) overrides the top-level depth."
        ),
    )


class MCTSBrainSettings(BaseBrainSettings):
    """Settings for the MCTS brain."""

    iterations: int = Field(default=10_000, ge=1, le=1_000_000)
    exploration: float = Field(default=2.0, ge=0.1, le=5.0)
    dirichlet_alpha: float = Field(default=0.03, ge=0.0, le=1.0)
    dirichlet_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    vcf_depth: int = Field(default=10, ge=0, le=40)


# ── Recording settings ──────────────────────────────────────────────────────


class EvaluationSettings(BaseModel):
    """Settings for the board evaluation function."""

    use_jit: bool = Field(default=True, description="Use Numba JIT-compiled evaluation (faster but requires numba).")


class BitboardSettings(BaseModel):
    """Settings for the bitboard representation."""

    enabled: bool = Field(default=False, description="Use row-based uint32 bitboard representation (Numba JIT).")


class RecordingSettings(BaseModel):
    """Settings for game recording."""

    enabled: bool = False
    directory: Path = Path("recordings")


class DiagnosticsSettings(BaseModel):
    """Settings for per-move diagnostics recording."""

    enabled: bool = False
    directory: Path = Path("recordings")
    top_candidates: int = Field(default=10, ge=1, le=50)


# ── Game-level settings ─────────────────────────────────────────────────────


class GameSettings(BaseModel):
    """Game configuration (board geometry)."""

    board_size: int = Field(default=15, ge=5, le=25)
    win_length: int = Field(default=5, ge=3)


# ── HTTP API settings ───────────────────────────────────────────────────────


class ApiSettings(BaseModel):
    """Settings that only apply to the ``gomoku-fastapi`` service.

    Lives in the core ``gomoku`` package so it sits alongside the other
    sections in ``gomoku.toml`` — no separate settings file needed.
    The CLI / engine ignore it; only ``gomoku_fastapi`` reads it.
    """

    cors_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Browser origins allowed to call the API.  Empty list = no "
            "CORS middleware mounted (server-to-server only).  Use "
            '``["*"]`` for fully open CORS, or list specific origins '
            'like ``["http://localhost:5173"]``.'
        ),
    )
    rate_limit_default: str = Field(
        default="120/minute",
        description=(
            "Default per-IP slowapi limit applied to all endpoints.  "
            'Format: ``"<count>/<period>"`` — see slowapi docs.  '
            "Set to empty string to disable the default limit."
        ),
    )
    rate_limit_compute: str = Field(
        default="30/minute",
        description=(
            "Tighter limit for CPU-heavy endpoints (``/v1/move`` and "
            "``/v1/games/{id}/think``) — overrides the default for "
            "those routes only.  Set empty to fall back to the default."
        ),
    )
    max_time_limit_s: float = Field(
        default=10.0,
        ge=0.0,
        le=600.0,
        description=(
            "Upper bound, in seconds, on the wall-clock search budget of a single "
            "API request.  Applied to ``time_limit_s`` of alpha-beta brains (an "
            "unlimited ``0`` becomes this value), to the alpha-beta fallback of "
            "the threat brain and to its ``pns_time_s``.  Brain overrides cannot "
            "exceed it.  ``0`` disables the cap.  The CLI is unaffected."
        ),
    )
    logfire_send: bool = Field(
        default=False,
        description=(
            "Send logs / spans to logfire.dev.  Requires "
            "``LOGFIRE_TOKEN`` env var; off by default so a fresh "
            "checkout doesn't emit network traffic."
        ),
    )
    logfire_console: bool = Field(
        default=True,
        description="Print structured logs to stderr.  Independent of ``logfire_send``.",
    )


# ── Brain settings container ────────────────────────────────────────────────


class BrainSettings(BaseModel):
    """Container holding brain alias configurations.

    Each alias is a ``[brains.<name>]`` TOML section with a ``type`` key
    referencing a built-in brain type and optional parameter overrides.
    Names that match a built-in type fall back to defaults when no alias
    is defined.

    Resolution lives in :func:`gomoku_brain.resolve_brain` because it
    needs the single brain registry.  This class stays a pure data
    container so the core ``gomoku`` package has no upward dependency.
    """

    model_config = ConfigDict(extra="allow")

    def aliases(self) -> dict[str, object]:
        """Return the alias table (``{alias_name: alias_dict}``).

        Brain *types* (the built-in names) are not included — they live
        in the registry alongside the brain classes.
        """
        return dict(self.model_extra or {})


# ── Root settings ────────────────────────────────────────────────────────────


class GomokuSettings(BaseSettings):
    """Root settings loaded from env vars / ``gomoku.toml`` / defaults.

    The TOML file is located dynamically on each instantiation via
    :func:`find_config_file` — meaning ``GomokuSettings()`` works the
    same whether you run from the project root or any subdirectory.
    Pre-walk-up code paths that constructed ``GomokuSettings(**data)``
    after loading TOML by hand continue to work unchanged because the
    discovered file is just a lower-priority source than ``init_settings``.
    """

    model_config = SettingsConfigDict(
        env_prefix="GOMOKU_",
        env_nested_delimiter="__",
    )

    game: GameSettings = GameSettings()
    brains: BrainSettings = BrainSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    bitboard: BitboardSettings = BitboardSettings()
    recording: RecordingSettings = RecordingSettings()
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()
    api: ApiSettings = ApiSettings()

    @override
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: object,
        env_settings: object,
        dotenv_settings: object,
        file_secret_settings: object,
    ) -> tuple[object, ...]:
        # Resolve the TOML path *per-instance* so subclasses, tests,
        # and CLI invocations from different CWDs all pick up the
        # nearest ``gomoku.toml``.  When no config is found, we still
        # construct a TomlConfigSettingsSource pointing at a missing
        # path — pydantic-settings handles that gracefully (it returns
        # an empty dict and field defaults are used).
        toml_path = find_config_file() or Path(CONFIG_FILENAME)
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=toml_path),
            file_secret_settings,
        )
