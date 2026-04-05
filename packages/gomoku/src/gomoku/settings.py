"""Gomoku settings — pydantic-settings based configuration.

Settings are loaded with the following priority (highest first):

1. Explicit ``__init__`` kwargs (programmatic overrides / CLI)
2. Environment variables with ``GOMOKU_`` prefix
3. ``gomoku.toml`` file in the current directory (if it exists)
4. Field defaults
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

_TOML_FILE = Path("gomoku.toml")


# ── Per-brain configuration sections ────────────────────────────────────────


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


class ThreatBrainSettings(BaseModel):
    """Settings for the threat brain."""

    depth: int = Field(default=4, ge=1, le=10)
    vcf_depth: int = Field(default=20, ge=1, le=40)
    vct_depth: int = Field(default=12, ge=1, le=30)
    threat_search: str = Field(default="pns", pattern="^(pns|vct)$")
    pns_nodes: int = Field(default=10_000, ge=1_000, le=10_000_000)
    pns_time_s: float = Field(default=5.0, ge=0.1, le=60.0)


# ── Recording settings ──────────────────────────────────────────────────────


class RecordingSettings(BaseModel):
    """Settings for game recording."""

    enabled: bool = False
    directory: Path = Path("recordings")


# ── Game-level settings ─────────────────────────────────────────────────────


class GameSettings(BaseModel):
    """Game configuration (board geometry)."""

    board_size: int = Field(default=15, ge=5, le=25)
    win_length: int = Field(default=5, ge=3)


# ── Brain settings container ────────────────────────────────────────────────


class BrainSettings(BaseModel):
    """Container holding per-brain configuration sections."""

    random: RandomBrainSettings = RandomBrainSettings()
    minimax: MinimaxBrainSettings = MinimaxBrainSettings()
    negamax: NegamaxBrainSettings = NegamaxBrainSettings()
    alphabeta: AlphaBetaBrainSettings = AlphaBetaBrainSettings()
    threat: ThreatBrainSettings = ThreatBrainSettings()


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
    recording: RecordingSettings = RecordingSettings()

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
