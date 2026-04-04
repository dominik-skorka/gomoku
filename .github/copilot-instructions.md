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
- Use `uv run` to execute project scripts; for example `uv run kimbler-cli …`.

## Workflow

Before making any changes, **summarise your understanding** of the request:

1. Restate the goal in your own words.
2. List the files you plan to modify or create.
3. Outline the steps you will take.
4. Wait for confirmation before proceeding.

This applies to every non-trivial prompt (code changes, refactors, new features, bug fixes). Trivial questions or read-only lookups do not require confirmation.

- Treat requested fixes, root-cause claims, and prescribed implementation details as hypotheses to verify, not facts to accept blindly.
- Briefly consider plausible alternatives and their trade-offs before choosing an approach.

- **Before / after:** When a change alters existing behaviour, describe how it worked before and how it works now. Use a table when comparing multiple attributes side-by-side; use bullet points when a single aspect changes or a table would be sparse.
- **Pros, cons, trade-offs:** For every non-trivial solution, briefly list the advantages, disadvantages, and key trade-offs so the decision rationale is visible.
- **Validation ownership:** After implementing a new solution or bug fix, validate the change yourself by default instead of asking the user to verify it.
- **Use tools and commands for verification:** When validation would benefit from available tools, tests, linters, sanity checks, HTTP calls, or other commands, run them yourself when it is safe and practical.
- **Ask before proceeding:** When a step requires user input, credentials, or access to external systems, ask for the necessary information and permission before proceeding.
- **Be transparent about limitations:** If you encounter a situation where you are uncertain, lack information, or face a limitation, be transparent about it and ask for guidance instead of making assumptions.
- **Be transparent about confidence:** If you are confident in a solution, say so. If you are uncertain, say that instead and explain the source of uncertainty.
- **Be transparent about assumptions:** If you make an assumption, state it explicitly. If the assumption turns out to be incorrect, acknowledge it and adjust your approach accordingly.
- **Be transparent about trade-offs:** If you choose an approach that has trade-offs, explain what they are and why you made that choice.
- **Be transparent about alternatives:** If you considered multiple approaches, briefly describe the alternatives and why you did not choose them.
- **Be transparent about next steps:** After completing a task, briefly describe what you did and what the next steps are, if any.
- **Be transparent about changes:** If you make a change to the codebase, briefly describe what you changed and why.
- **Be transparent about testing:** If you run tests, linters, or other validation tools, briefly describe what you ran and what the results were.
