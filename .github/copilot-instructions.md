# Copilot Instructions — gomoku workspace

## Project overview

This is a **uv workspace**

## Environment

- **OS:** Ubuntu 24.04
- **Shell:** bash-compatible
- **Python:** 3.14
- **Package manager:** uv (workspace root `pyproject.toml`)
- **Build backend:** hatchling

## Coding rules

- Prefer **uv** over pip for all package operations (`uv add`, `uv run`, `uv sync`, etc.).
- Prefer **pydantic** for data modelling and validation.
- Prefer **pydantic-settings** for application settings and configuration management.
- Use **Pydantic Logfire** as the default logging and observability framework. Prefer a shared Logfire setup over ad hoc `print` statements or fragmented logging configuration, and instrument application entrypoints centrally.
- Use **ruff** for linting and formatting (`uv run ruff check .` / `uv run ruff format .`).
- Use **pytest** for tests (`uv run pytest`).
- Use **git** as the recommended solution for source management.
- Keep diffs **minimal and localized** — touch only what is necessary.
- Review each new change for significant code duplication. If a change introduces or reveals substantial duplication, recommend considering a better solution and ask before proceeding with that approach.
- Do **not** introduce new dependencies without clear justification.
- If a new tool or package is required, use the latest available **stable** version.

## Shell commands

- Provide copy-pasteable commands.
- Assume bash-compatible shell.
- Use `uv run` to execute project scripts; for example `uv run gomoku-cli …`.

## Workflow

### Before implementation

Before making any changes, **summarise your understanding** of the request:

1. Restate the goal in your own words.
2. List the files you plan to modify or create.
3. Outline the steps you will take.
4. Wait for confirmation before proceeding.

This applies to every non-trivial prompt (code changes, refactors, new features, bug fixes). Trivial questions or read-only lookups do not require confirmation.

- Treat requested fixes, root-cause claims, and prescribed implementation details as hypotheses to verify, not facts to accept blindly.
- For non-trivial solutions, briefly consider plausible alternatives and their trade-offs before choosing an approach.

### After implementation

- **Validate changes yourself** using tests, linters, sanity checks, or other available tools instead of asking the user to verify. Ask only when a step requires user input, credentials, or access to external systems.
- **Describe what changed:** When a change alters existing behaviour, explain the before/after. Use a table when comparing multiple attributes side-by-side; use bullet points when a single aspect changes or a table would be sparse.

### Communication

Be transparent throughout. In particular:

- State your confidence level (confident vs uncertain) and the source of any uncertainty.
- Make assumptions explicit; acknowledge and adjust when they turn out to be wrong.
- When multiple approaches were considered, briefly note the alternatives and why you chose the one you did.
- Briefly summarise what was done, what changed, and any remaining next steps.
