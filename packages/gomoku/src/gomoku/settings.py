"""Gomoku settings — pydantic-settings based configuration.

Settings are loaded with the following priority (highest first):

1. Explicit ``__init__`` kwargs (programmatic overrides / CLI)
2. Environment variables with ``GOMOKU_`` prefix
3. ``gomoku.toml`` file in the current directory (if it exists)
4. Field defaults

Per-brain settings models are defined here, but the mapping from brain
name to settings model lives alongside the brain classes themselves in
``gomoku_brain.BRAINS`` (see :mod:`gomoku_brain` for the single registry).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

_TOML_FILE = Path("gomoku.toml")


# ── Per-brain configuration sections ────────────────────────────────────────


class KeyboardBrainSettings(BaseModel):
    """Settings for the keyboard (human) provider (no tunables)."""


class RandomBrainSettings(BaseModel):
    """Settings for the random brain (no tunables)."""


class MinimaxBrainSettings(BaseModel):
    """Settings for the minimax brain."""

    depth: int = Field(default=3, ge=1, le=10)


class NegamaxBrainSettings(BaseModel):
    """Settings for the negamax brain."""

    depth: int = Field(default=3, ge=1, le=10)


class AlphaBetaBrainSettings(BaseModel):
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
    time_limit_s: float = Field(default=0.0, ge=0.0, le=600.0)
    opening_book: str = Field(default="", description="Path to an opening book JSON file.")


class ThreatBrainSettings(BaseModel):
    """Settings for the threat brain."""

    depth: int = Field(default=4, ge=1, le=10)
    vcf_depth: int = Field(default=20, ge=1, le=40)
    vct_depth: int = Field(default=12, ge=1, le=30)
    threat_search: str = Field(default="pns", pattern="^(pns|vct)$")
    pns_nodes: int = Field(default=10_000, ge=1_000, le=10_000_000)
    pns_time_s: float = Field(default=5.0, ge=0.1, le=60.0)


class MCTSBrainSettings(BaseModel):
    """Settings for the MCTS brain."""

    iterations: int = Field(default=10_000, ge=100, le=1_000_000)
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
    """Root settings loaded from env vars / ``gomoku.toml`` / defaults."""

    model_config = SettingsConfigDict(
        env_prefix="GOMOKU_",
        env_nested_delimiter="__",
        toml_file=str(_TOML_FILE),
    )

    game: GameSettings = GameSettings()
    brains: BrainSettings = BrainSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    bitboard: BitboardSettings = BitboardSettings()
    recording: RecordingSettings = RecordingSettings()
    diagnostics: DiagnosticsSettings = DiagnosticsSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: object,
        env_settings: object,
        dotenv_settings: object,
        file_secret_settings: object,
    ) -> tuple[object, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
