/**
 * Difficulty levels shown in the UI, each a brain plus per-request overrides
 * sent to `POST /v1/move`.  Brain names come from `GET /v1/brains`; the
 * server caps search time (`[api].max_time_limit_s`) whatever we send.
 *
 * Think times are rough, measured on a laptop-class CPU in the middle game.
 */

export type PresetId = "easy" | "medium" | "hard" | "master";

export interface Preset {
  readonly id: PresetId;
  readonly brain: string;
  readonly overrides?: Record<string, unknown>;
  /** Typical think time, for the UI only. */
  readonly typicalSeconds: number;
}

export const PRESETS: readonly Preset[] = [
  { id: "easy", brain: "negamax", overrides: { depth: 2 }, typicalSeconds: 0.5 },
  { id: "medium", brain: "alphabeta", overrides: { depth: 2, vcf_depth: 0 }, typicalSeconds: 0.5 },
  { id: "hard", brain: "alphabeta", typicalSeconds: 1 },
  { id: "master", brain: "threat", typicalSeconds: 6 },
];

export const DEFAULT_PRESET: PresetId = "hard";

export function presetById(id: PresetId): Preset {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) throw new Error(`unknown preset ${id}`);
  return preset;
}
