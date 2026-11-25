import createClient from "openapi-fetch";
import type { Cell, Grid, Player } from "../game/rules";
import type { components, paths } from "./schema";

/**
 * Typed client for the Gomoku REST API.  Types are generated from
 * `web/openapi/v1.json` (`pnpm gen:api`), which CI keeps in sync with the
 * server, so an incompatible API change breaks the web build, not the game.
 *
 * The base URL comes from `apiBase` in config.json (`configureApi`, called
 * at start-up); the default `/api` is same-origin: Caddy (production) and
 * the Vite dev server both proxy it to the API, so no CORS is involved.
 */
export let api = createClient<paths>({ baseUrl: "/api" });

/** Point the client at the API (`apiBase` from config.json). */
export function configureApi(baseUrl: string): void {
  api = createClient<paths>({ baseUrl });
}

export type ApiErrorKind = "rate_limited" | "unavailable" | "rejected";

export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    readonly status: number | undefined,
    message: string,
    readonly requestId?: string | null,
    /** Seconds to wait before retrying, from a 429's `Retry-After` header. */
    readonly retryAfterS?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface EngineMove {
  readonly cell: Cell;
  readonly brain: string;
  readonly elapsedMs: number;
  readonly requestId: string | null;
}

export async function requestMove(
  args: {
    grid: Grid;
    sideToMove: Player;
    brain: string;
    overrides?: Record<string, unknown>;
  },
  signal?: AbortSignal,
): Promise<EngineMove> {
  const body: components["schemas"]["MoveRequest"] = {
    board: { size: args.grid.length, grid: args.grid.map((row) => [...row]) },
    side_to_move: args.sideToMove,
    brain: args.brain,
    brain_overrides: args.overrides ?? null,
  };
  const started = performance.now();
  const result = await api.POST("/v1/move", { body, signal }).catch((err: unknown) => {
    if (signal?.aborted) throw err;
    throw new ApiError("unavailable", undefined, "The game server cannot be reached.");
  });
  const { data, response } = result;
  const requestId = response.headers.get("X-Request-ID");
  if (!data) {
    const kind: ApiErrorKind =
      response.status === 429 ? "rate_limited" : response.status >= 500 ? "unavailable" : "rejected";
    const retryAfter = Number.parseInt(response.headers.get("Retry-After") ?? "", 10);
    throw new ApiError(
      kind,
      response.status,
      `The game server answered ${response.status}.`,
      requestId,
      Number.isFinite(retryAfter) ? retryAfter : undefined,
    );
  }
  return {
    cell: { row: data.move.row, col: data.move.col },
    brain: data.brain,
    elapsedMs: performance.now() - started,
    requestId,
  };
}
