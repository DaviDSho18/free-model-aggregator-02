// routing.ts: quality-first model choice, per-model cooldowns, and failover planning.
//
// The score's biggest column (quality, 60%) is the sealed rating of the model that REALLY served each
// prompt, over the best free model that meets that prompt's requirements. So the choice is not "the
// first model that fits": it is the highest-rated model that fits, and every prompt's requirement set
// has its own best. This module computes, per requirement shape, the ranked candidate list once, then
// hands out candidates in that order while a prompt is being served.
//
// Everything here is a pure function of the brief: no wall clock, no unseeded randomness, so a live
// pass and its replay pick the same model for the same prompt.

import {
  capabilitiesOf,
  freeModels,
  meetsRequirements,
  type Brief,
  type ModelRow,
  type Requirement,
} from './providers.ts';

/** A requirement set as a stable string: candidates are ranked once per distinct shape. */
export function requirementKey(requires: Requirement): string {
  return [
    requires.tools === true ? 't' : '-',
    requires.vision === true ? 'v' : '-',
    requires.json === true ? 'j' : '-',
    requires.min_context === undefined ? '0' : String(requires.min_context),
  ].join('');
}

/** The free models that can serve one requirement set, best rating first, ties by the brief's order. */
export function candidatesFor(brief: Brief, requires: Requirement): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    out.push(model);
  }
  // A stable sort: equal ratings keep the brief's order, so the choice never depends on the engine's
  // sort stability. Higher rating_milli first.
  return out
    .map((model, index) => ({ model, index }))
    .sort((a, b) => (b.model.rating_milli - a.model.rating_milli) || (a.index - b.index))
    .map((row) => row.model);
}

/** The best free rating available for one requirement set: the denominator the quality column uses. */
export function bestRatingFor(brief: Brief, requires: Requirement): number {
  const candidates = candidatesFor(brief, requires);
  return candidates.length === 0 ? 0 : candidates[0].rating_milli;
}

/**
 * Every requirement shape the brief declares, ranked once. A request whose shape the brief does not
 * declare (a brief that holds no prompt rows) is ranked on demand and remembered.
 */
export class CandidateTable {
  private readonly brief: Brief;
  private readonly byKey = new Map<string, ModelRow[]>();

  constructor(brief: Brief) {
    this.brief = brief;
    for (const prompt of brief.prompts) {
      const key = requirementKey(prompt.requires);
      if (!this.byKey.has(key)) this.byKey.set(key, candidatesFor(brief, prompt.requires));
    }
  }

  /** The ranked candidates for one requirement set, computed once. */
  candidates(requires: Requirement): ModelRow[] {
    const key = requirementKey(requires);
    let list = this.byKey.get(key);
    if (list === undefined) {
      list = candidatesFor(this.brief, requires);
      this.byKey.set(key, list);
    }
    return list;
  }
}

/**
 * Per-model cooldowns. A provider's 429 or 5xx is real and reaches this server as the provider sent it:
 * nothing retries it. A model that refused is not asked again while it cools, so one rate-limited model
 * cannot take every later prompt down with it. The cooldown is counted in CALLS, never in wall-clock
 * time: a trial is replayed, and a decision that read the clock would not reproduce.
 */
export class Cooldowns {
  /** model id -> calls still to skip before it is tried again. */
  private readonly remaining = new Map<string, number>();
  private readonly step: number;
  private readonly max: number;

  constructor(step = 2, max = 8) {
    this.step = step;
    this.max = max;
  }

  /** One prompt has been served: every model's cooldown has had its one call of rest. */
  advance(): void {
    for (const [id, left] of [...this.remaining]) {
      if (left <= 1) this.remaining.delete(id);
      else this.remaining.set(id, left - 1);
    }
  }

  /** A model refused: sit it out for a while, longer each time it refuses again. */
  penalise(id: string): void {
    const left = this.remaining.get(id) ?? 0;
    this.remaining.set(id, Math.min(this.max, left + this.step));
  }

  cooling(id: string): boolean {
    return (this.remaining.get(id) ?? 0) > 0;
  }
}

/**
 * The models to try for one prompt, in order: the ranked candidates with the cooling ones moved to the
 * back (never dropped — if every candidate is cooling, the prompt is still worth a call). The list is
 * capped so a trial cannot spend its whole 60-call ceiling on one prompt.
 */
export function planFor(candidates: readonly ModelRow[], cooldowns: Cooldowns, cap = 4): ModelRow[] {
  const warm: ModelRow[] = [];
  const cold: ModelRow[] = [];
  for (const model of candidates) (cooldowns.cooling(model.id) ? cold : warm).push(model);
  return [...warm, ...cold].slice(0, cap);
}

/**
 * One gateway refusal, as the optional `x-makefun-upstream-failures` trail reports it. The trail is the
 * only way the failover column can see a refusal this server retried and then served: the env sees the
 * answers, never the gateway's traffic.
 */
export interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

/** The refusal trail as the header's JSON array: at most 8 rows, in the order the calls refused. */
export function refusalTrail(refusals: readonly Refusal[]): string {
  return JSON.stringify(refusals.slice(0, 8).map((r) => ({ status: r.status, code: r.code, model: r.model })));
}

/**
 * The machine-readable code of a refusal: the provider's own error code when its body carries one, else
 * the HTTP reason as a short snake_case word. The column reads the row's shape, never its meaning.
 */
export function refusalCode(status: number, body: string): string {
  const named = codeIn(body);
  if (named !== null) return named;
  if (status === 429) return 'rate_limited';
  if (status === 401 || status === 403) return 'unauthorized';
  if (status === 402) return 'payment_required';
  if (status === 400) return 'invalid_request';
  if (status === 404) return 'not_found';
  if (status >= 500) return 'upstream_error';
  return 'refused';
}

/** The first `code` string in a provider's error body, or null. */
function codeIn(body: string): string | null {
  if (body === '') return null;
  try {
    const parsed: unknown = JSON.parse(body);
    if (parsed === null || typeof parsed !== 'object') return null;
    const root = parsed as Record<string, unknown>;
    const inner = root.error;
    if (inner !== null && typeof inner === 'object') {
      const code = (inner as Record<string, unknown>).code;
      if (typeof code === 'string' && code !== '') return code;
    }
    if (typeof root.code === 'string' && root.code !== '') return root.code;
  } catch {
    // Not JSON: no code to read.
  }
  return null;
}
