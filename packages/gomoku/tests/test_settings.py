"""Tests for ``gomoku.settings`` config discovery.

Before the walk-up patch, ``gomoku.toml`` was looked up by passing the
literal string ``"gomoku.toml"`` to pydantic-settings, which resolves
relative to the current working directory.  Running ``gomoku ...``
from any subdirectory of the project silently dropped the config and
fell back to defaults — exactly the kind of silent-misconfiguration
that's hard to spot.

These tests pin three layers of the new behaviour:

* :func:`find_config_file` walks up correctly, respects project-root
  markers, and honours the ``GOMOKU_CONFIG`` env var override.
* :class:`GomokuSettings` instantiated from a subdirectory picks up
  the parent's ``gomoku.toml`` automatically.
* When no config exists anywhere in the search path, the settings
  fall back to field defaults without raising.
"""

from gomoku.settings import (
    CONFIG_FILENAME,
    GomokuSettings,
    find_config_file,
)

# ---------------------------------------------------------------------------
# find_config_file — walk-up logic
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    def test_finds_in_current_directory(self, tmp_path, monkeypatch):
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text("[game]\nboard_size = 13\n")
        monkeypatch.chdir(tmp_path)
        assert find_config_file() == cfg.resolve()

    def test_walks_up_from_subdirectory(self, tmp_path, monkeypatch):
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text("[game]\nboard_size = 13\n")
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        assert find_config_file() == cfg.resolve()

    def test_returns_none_when_no_config_in_any_ancestor(self, tmp_path, monkeypatch):
        # tmp_path is empty, no config anywhere up the tree below the
        # filesystem root.  We don't go all the way to ``/`` — the
        # project-root markers stop us at the temp's parent.
        deep = tmp_path / "deep"
        deep.mkdir()
        # Add a project-root marker at tmp_path so the walk-up stops
        # cleanly without scanning the whole filesystem.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        monkeypatch.chdir(deep)
        assert find_config_file() is None

    def test_stops_at_project_root_marker(self, tmp_path, monkeypatch):
        """A ``pyproject.toml`` in a directory that does *not* contain
        ``gomoku.toml`` is a project boundary — the walk-up must not
        leak into a parent project."""
        # Outer directory has gomoku.toml — should NOT be picked.
        (tmp_path / CONFIG_FILENAME).write_text("# parent-config")
        # Inner project has pyproject.toml but no gomoku.toml.
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\nname='child'\n")
        sub = project / "src"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert find_config_file() is None

    def test_pyproject_alongside_gomoku_returns_gomoku(self, tmp_path, monkeypatch):
        """When both markers are in the same directory, the gomoku.toml
        wins — the project-root check only fires when *no* gomoku.toml
        is present at that level."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text("[game]\nboard_size = 13\n")
        sub = tmp_path / "src"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert find_config_file() == cfg.resolve()

    def test_git_dir_also_acts_as_root_marker(self, tmp_path, monkeypatch):
        # ``.git`` is checked alongside ``pyproject.toml`` — useful when
        # the project doesn't use a Python ``pyproject.toml`` (e.g.,
        # contributors who clone but haven't run ``uv sync``).
        outer_cfg = tmp_path / CONFIG_FILENAME
        outer_cfg.write_text("# outer")
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        sub = repo / "src"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert find_config_file() is None

    def test_env_var_override_absolute_path(self, tmp_path, monkeypatch):
        cfg = tmp_path / "custom.toml"
        cfg.write_text("[game]\nboard_size = 13\n")
        # CWD is irrelevant — env var wins.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GOMOKU_CONFIG", str(cfg))
        assert find_config_file() == cfg

    def test_env_var_override_missing_file_returns_none(self, tmp_path, monkeypatch):
        """When the user explicitly sets ``GOMOKU_CONFIG`` to a path
        that doesn't exist, return ``None`` — *don't* fall through to
        the walk-up.  Falling through would silently pick a different
        config than the user asked for."""
        # Set up a walk-up config that should NOT be picked.
        (tmp_path / CONFIG_FILENAME).write_text("# walk-up config")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GOMOKU_CONFIG", str(tmp_path / "missing.toml"))
        assert find_config_file() is None

    def test_empty_env_var_falls_through_to_walk_up(self, tmp_path, monkeypatch):
        """An empty ``GOMOKU_CONFIG`` is treated as unset — otherwise
        a stale ``GOMOKU_CONFIG=`` in the shell would suppress all
        config discovery."""
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text("[game]\nboard_size = 13\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GOMOKU_CONFIG", "")
        assert find_config_file() == cfg.resolve()


# ---------------------------------------------------------------------------
# GomokuSettings — end-to-end integration
# ---------------------------------------------------------------------------


class TestGomokuSettingsLoadsDiscoveredConfig:
    def test_loads_config_from_parent_directory(self, tmp_path, monkeypatch):
        """The whole point of the walk-up: running from a subdirectory
        must pick up the project's config."""
        (tmp_path / CONFIG_FILENAME).write_text("[game]\nboard_size = 13\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        # Clear any inherited env var from the parent process.
        monkeypatch.delenv("GOMOKU_CONFIG", raising=False)
        s = GomokuSettings()
        assert s.game.board_size == 13

    def test_falls_back_to_defaults_when_no_config(self, tmp_path, monkeypatch):
        # Marker at tmp_path so the walk-up stops cleanly without
        # scanning the developer's home directory.
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        monkeypatch.delenv("GOMOKU_CONFIG", raising=False)
        s = GomokuSettings()
        assert s.game.board_size == 15  # field default

    def test_env_var_takes_precedence_over_walk_up(self, tmp_path, monkeypatch):
        # Walk-up would find the close one (board_size=13)…
        (tmp_path / CONFIG_FILENAME).write_text("[game]\nboard_size = 13\n")
        # …but the env var points at the explicit one (board_size=11).
        explicit = tmp_path / "explicit.toml"
        explicit.write_text("[game]\nboard_size = 11\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GOMOKU_CONFIG", str(explicit))
        s = GomokuSettings()
        assert s.game.board_size == 11
