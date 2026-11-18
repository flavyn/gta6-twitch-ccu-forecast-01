// gta6-twitch-ccu-forecast-01 — launch-day Twitch concurrent-viewer forecast.
//
// The grader hands this program one case file: everything the house recorded up to the end of
// 2026-11-19's first UTC hour, hour 1's print included. It must answer with five quantiles for each
// of hours 2..24. The score is the mean pinball loss on the log scale, so only the *shape* of the
// curve (ratios to hour 1) and the *spread* around it matter; the absolute audience size cancels out.
//
// Method (the numbers behind it are in SPEC.md beside the kit, measured on the four practice
// launches p00..p03): every comparable launch is turned into a 24-slot hour-of-day profile relative
// to its own release day. For each hour h we collect r = ln(profile[h-1] / profile[0]) over the
// comparables that have both slots, weight each comparable by recency and by how close its hour-1
// level was to this case's, and take the weighted median as the centre. A weighted MAD around that
// centre, scaled by 0.7, gives the log-scale spread that turns the centre into five quantiles.
//
// Everything here is deterministic: the same case file always produces the same bytes, which the
// grader requires of its two runs.
//
// Local run: MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

/** Where the case files are read from and the forecasts written to. */
const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The five levels the contest scores, in the order each answer row lists them. */
const TAU = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each of those levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Hours 2..24 are forecast; hour 1 is given. */
const FIRST_HOUR = 2;
const LAST_HOUR = 24;
/** A viewer count never leaves (0, 1e9]. */
const MAX_VIEWERS = 1e9;

/** Reference year the recency weight counts back from, and its half-life in years. */
const REFERENCE_YEAR = 2026;
const HALF_LIFE_YEARS = 2;
/** How hard a comparable's hour-1 level is pulled away from this case's own. */
const LEVEL_WEIGHT = 0.35;
/** Multiplier and floor of the log-scale spread. */
const SIGMA_SCALE = 0.7;
const SIGMA_FLOOR = 0.12;
/** A comparable needs at least this many viewers at hour 1 to be usable at all. */
const MIN_HOUR1_VIEWERS = 200;
/** ... and at least this many of the other 23 slots present. */
const MIN_SLOTS_PRESENT = 20;

/**
 * Median log-ratio shape, ln(viewers_h / viewers_1) for h = 2..24, measured over the comparable
 * launches. Used only when a case leaves too few comparable observations for an hour.
 */
const FALLBACK_SHAPE = [
  -0.132, -0.28, -0.307, -0.282, -0.25, -0.226, -0.155, -0.138, -0.128, -0.099, -0.028, 0.052,
  0.147, 0.375, 0.483, 0.519, 0.602, 0.797, 0.836, 0.829, 0.805, 0.761, 0.678,
];

interface Point {
  from: string;
  viewers: number | null;
  channels?: number | null;
}

interface Comparable {
  name?: string;
  day_start: string;
  hourly?: Point[] | null;
}

interface CaseFile {
  id: string;
  day_start: string;
  levels?: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast?: Array<{ hour: number }>;
  comparables?: Comparable[] | null;
}

/** One comparable reduced to what the model needs: a level per hour of its release day. */
interface Sample {
  /** Viewers at each hour of the day; null where the source had no point for that hour. */
  profile: Array<number | null>;
  /** The weight this sample carries in the medians. */
  weight: number;
}

/** Weighted median of (value, weight) pairs; the first value whose weight reaches half the total. */
function weightedMedian(pairs: Array<[number, number]>): number {
  const sorted = [...pairs].sort((a, b) => a[0] - b[0]);
  let total = 0;
  for (const [, w] of sorted) total += w;
  let running = 0;
  for (const [value, w] of sorted) {
    running += w;
    if (running >= total / 2) return value;
  }
  return sorted[sorted.length - 1][0];
}

/** Weighted median absolute deviation around a already-computed weighted median. */
function weightedMad(pairs: Array<[number, number]>, centre: number): number {
  return weightedMedian(pairs.map(([value, w]) => [Math.abs(value - centre), w]));
}

/** The hour of the release day a timestamp falls in, as an index 0..23. */
function slotOf(timestamp: string, dayStart: number): number {
  const t = Date.parse(timestamp);
  if (!Number.isFinite(t)) return -1;
  return Math.floor((t - dayStart) / 3_600_000);
}

/**
 * Reduce one comparable to a 24-slot profile of mean viewers per hour of its release day, plus the
 * weight it should carry when this case's hour-1 print and its own are compared.
 */
function toSample(comp: Comparable, dayStart: number, p1: number): Sample | null {
  const start = Date.parse(comp.day_start);
  const points = comp.hourly;
  if (!Number.isFinite(start) || !Array.isArray(points)) return null;

  const sums = new Array<number>(24).fill(0);
  const counts = new Array<number>(24).fill(0);
  for (const point of points) {
    const slot = slotOf(point.from, start);
    if (slot < 0 || slot > 23) continue;
    const viewers = point.viewers;
    if (typeof viewers !== 'number' || !Number.isFinite(viewers) || viewers <= 0) continue;
    sums[slot] += viewers;
    counts[slot] += 1;
  }

  const profile: Array<number | null> = sums.map((sum, slot) => (counts[slot] > 0 ? sum / counts[slot] : null));
  const hour1 = profile[0];
  if (hour1 === null || hour1 < MIN_HOUR1_VIEWERS) return null;

  let present = 0;
  for (let slot = 1; slot < 24; slot++) if (profile[slot] !== null) present += 1;
  if (present < MIN_SLOTS_PRESENT) return null;

  const year = new Date(comp.day_start).getUTCFullYear();
  const recency = Number.isFinite(year) ? Math.exp(-(REFERENCE_YEAR - year) / HALF_LIFE_YEARS) : 1;
  const level = Math.exp(-LEVEL_WEIGHT * Math.abs(Math.log(hour1) - Math.log(p1)));
  return { profile, weight: recency * level };
}

/** Five viewer quantiles for every forecast hour of one case. */
function forecastCase(caseFile: CaseFile): Array<{ hour: number; q: number[] }> {
  const known = caseFile.known.find((row) => row.hour === 1);
  const p1 = Math.max(1, Number.isFinite(known?.viewers) ? Number(known?.viewers) : 1);
  const dayStart = Date.parse(caseFile.day_start);

  // Every usable comparable, with its weight already folded in.
  const samples: Sample[] = [];
  for (const comp of caseFile.comparables ?? []) {
    if (!comp || typeof comp !== 'object') continue;
    const sample = toSample(comp, dayStart, p1);
    if (sample !== null) samples.push(sample);
  }

  const hours: number[] = [];
  for (let hour = FIRST_HOUR; hour <= LAST_HOUR; hour++) hours.push(hour);
  if (samples.length === 0) {
    return hours.map((hour) => ({
      hour,
      q: Z.map((z) => roundQuantile(p1 * Math.exp(FALLBACK_SHAPE[hour - 2] + z * 0.35))),
    }));
  }

  return hours.map((hour) => {
    const slot = hour - 1;
    const observations: Array<[number, number]> = [];
    for (const sample of samples) {
      const value = sample.profile[slot];
      const base = sample.profile[0];
      if (value === null || base === null || value <= 0 || base <= 0) continue;
      observations.push([Math.log(value / base), sample.weight]);
    }

    let centre: number;
    let spread: number;
    if (observations.length < 3) {
      centre = FALLBACK_SHAPE[hour - 2];
      spread = 0.35;
    } else {
      centre = weightedMedian(observations);
      spread = Math.max(SIGMA_FLOOR, SIGMA_SCALE * 1.4826 * weightedMad(observations, centre));
    }
    const q = Z.map((z) => roundQuantile(p1 * Math.exp(centre + z * spread)));
    return { hour, q: tidy(q) };
  });
}

/** A quantile the answer may carry: finite, above 0, at most 1e9, rounded to two decimals. */
function roundQuantile(value: number): number {
  if (!Number.isFinite(value)) return 1;
  const rounded = Math.round(value * 100) / 100;
  return Math.min(MAX_VIEWERS, Math.max(0.01, rounded));
}

/** Round five quantiles, then force them non-decreasing as the answer format requires. */
function tidy(q: number[]): number[] {
  const sorted = [...q].sort((a, b) => a - b);
  return sorted;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Name order keeps two runs on the same cases writing the same bytes.
const caseFiles = readdirSync(INPUT_DIR)
  .filter((name) => name.endsWith('.in'))
  .sort();

for (const fileName of caseFiles) {
  const id = fileName.slice(0, -'.in'.length);
  const caseFile = JSON.parse(readFileSync(join(INPUT_DIR, fileName), 'utf8')) as CaseFile;
  const hours = forecastCase(caseFile);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
