// gta6-twitch-ccu-forecast-01 — launch-day Twitch concurrent-viewer forecast.
//
// The grader hands this program one case file: everything the house recorded up to the end of
// 2026-11-19's first UTC hour, hour 1's print included. It must answer with five quantiles for each
// of hours 2..24. The score is the mean pinball loss on the log scale, so only the *shape* of the
// curve (ratios to hour 1) and the *spread* around it matter; the absolute audience size cancels out.
//
// Two models are built and averaged in log space, because they see different parts of the same fact.
//
//   A. the shape model. Every comparable launch becomes a 24-slot profile of mean viewers per hour
//      of its own release day. For each hour h it contributes r = ln(profile[h-1] / profile[0]),
//      weighted by recency and by how close its hour-1 level was to this case's. The weighted median
//      of those r values is the centre; a weighted MAD around it, scaled by 0.7, is the spread.
//
//   B. the level model. Hour 1 is one noisy reading of how much of the world was already watching;
//      the peak of the hours just before the release day is another. A launch whose hour 1 fell far
//      below that peak had caught only part of the wave and tends to keep ramping, while one whose
//      hour 1 already sat near the peak flattens. So for each comparable we regress its log level
//      ln(viewers_h / pre-release peak) on ln(hour1 / pre-release peak), weight each comparable by
//      how closely its pre-release trajectory matched this case's, and read the five quantiles off
//      the regression residuals.
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

const HOUR_MS = 3_600_000;
const FIRST_HOUR = 2;
const LAST_HOUR = 24;
const MAX_VIEWERS = 1e9;

/** Model A: the comparable-shape medians. */
const REFERENCE_YEAR = 2026;
const HALF_LIFE_YEARS = 2;
const LEVEL_WEIGHT = 0.35;
const SIGMA_SCALE = 0.7;
const SIGMA_FLOOR = 0.12;
const MIN_HOUR1_VIEWERS = 200;
const MIN_SLOTS_PRESENT = 20;

/** Model B: the pre-release-peak regression. */
const TRAIL_HOURS = 4;
const BANDWIDTH = 1;
const MIN_LEVEL_HOUR1 = 10_000;
const B_SPREAD_SCALE = 0.9;
const MIN_REGRESSION_ROWS = 8;

/** How much of the answer comes from model B, in log space. */
const BLEND_WEIGHT = 0.5;

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
  history_hourly?: Point[] | null;
  comparables?: Comparable[] | null;
}

/** A comparable reduced to what the models need. */
interface Sample {
  /** Mean viewers at each hour of its release day, null where the source had no point. */
  profile: Array<number | null>;
  /**
   * Viewers at each of the hours just before its release day, oldest first, or null when the
   * source has a gap there. Model A only needs the day itself, so a comparable with a gap is kept
   * for the shape medians and skipped by model B alone.
   */
  trail: number[] | null;
  /** Model A's weight. */
  weight: number;
}

/** log of a positive finite number, or null. */
function logOf(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.log(value) : null;
}

/** Weighted median: the first value whose running weight reaches half the total. */
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

/** Weighted median absolute deviation around a weighted median. */
function weightedMad(pairs: Array<[number, number]>, centre: number): number {
  return weightedMedian(pairs.map(([value, w]) => [Math.abs(value - centre), w]));
}

/** Linear-interpolated quantile of an unsorted list (type 7). */
function quantile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const x = (sorted.length - 1) * p;
  const lo = Math.floor(x);
  const frac = x - lo;
  return lo + 1 < sorted.length ? sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]) : sorted[sorted.length - 1];
}

/** Hour index of a timestamp relative to a release day, or -1 when unparsable. */
function hourIndex(timestamp: string, dayStartMs: number): number {
  const t = Date.parse(timestamp);
  return Number.isFinite(t) ? Math.floor((t - dayStartMs) / HOUR_MS) : -1;
}

/**
 * Reduce one comparable to its 24-slot day profile, its pre-release peak and trail, and the weight
 * model A should give it.
 */
function toSample(comp: Comparable, p1: number): Sample | null {
  const start = Date.parse(comp.day_start);
  const points = comp.hourly;
  if (!Number.isFinite(start) || !Array.isArray(points)) return null;

  const sums = new Array<number>(24).fill(0);
  const counts = new Array<number>(24).fill(0);
  const before = new Map<number, number>();
  for (const point of points) {
    const slot = hourIndex(point.from, start);
    const viewers = point.viewers;
    if (!Number.isFinite(slot) || slot < -TRAIL_HOURS) continue;
    if (typeof viewers !== 'number' || !Number.isFinite(viewers) || viewers <= 0) continue;
    if (slot < 0) {
      before.set(slot, viewers);
    } else if (slot <= 23) {
      sums[slot] += viewers;
      counts[slot] += 1;
    }
  }

  const profile: Array<number | null> = sums.map((sum, slot) => (counts[slot] > 0 ? sum / counts[slot] : null));
  const hour1 = profile[0];
  if (hour1 === null || hour1 < MIN_HOUR1_VIEWERS) return null;

  let present = 0;
  for (let slot = 1; slot < 24; slot++) if (profile[slot] !== null) present += 1;
  if (present < MIN_SLOTS_PRESENT) return null;

  const trail: number[] = [];
  let trailComplete = true;
  for (let slot = -TRAIL_HOURS; slot <= -1; slot++) {
    const value = before.get(slot);
    if (value === undefined) { trailComplete = false; break; }
    trail.push(value);
  }

  const year = new Date(comp.day_start).getUTCFullYear();
  const recency = Number.isFinite(year) ? Math.exp(-(REFERENCE_YEAR - year) / HALF_LIFE_YEARS) : 1;
  const level = Math.exp(-LEVEL_WEIGHT * Math.abs(Math.log(hour1) - Math.log(p1)));
  return {
    profile,
    trail: trailComplete && trail.length === TRAIL_HOURS ? trail : null,
    weight: recency * level,
  };
}

/** Model A: weighted median shape and weighted MAD spread. */
function shapeModel(samples: Sample[], hour: number): { centre: number; spread: number } {
  const slot = hour - 1;
  const observations: Array<[number, number]> = [];
  for (const sample of samples) {
    const value = sample.profile[slot];
    const base = sample.profile[0];
    if (value === null || base === null) continue;
    const logged = logOf(value);
    const baseLog = logOf(base);
    if (logged === null || baseLog === null) continue;
    observations.push([logged - baseLog, sample.weight]);
  }
  if (observations.length < 3) {
    return { centre: FALLBACK_SHAPE[hour - 2], spread: 0.35 };
  }
  const centre = weightedMedian(observations);
  const spread = Math.max(SIGMA_FLOOR, SIGMA_SCALE * 1.4826 * weightedMad(observations, centre));
  return { centre, spread };
}

/** Standardise a log trajectory to zero mean and unit spread, so shapes compare. */
function standardiseTrail(trail: number[]): number[] {
  const logs = trail.map(Math.log);
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length;
  const sd = Math.sqrt(logs.reduce((a, b) => a + (b - mean) ** 2, 0) / logs.length) || 1;
  return logs.map((value) => (value - mean) / sd);
}

/** The hours just before the release day, as hourly means, or null when any is missing. */
function preReleaseTrail(rows: Point[] | null | undefined, dayStartMs: number): number[] | null {
  if (!Array.isArray(rows)) return null;
  const byHour = new Map<number, number>();
  for (const row of rows) {
    const slot = hourIndex(row.from, dayStartMs);
    const viewers = row.viewers;
    if (viewers === null || typeof viewers !== 'number' || !Number.isFinite(viewers) || viewers <= 0) continue;
    if (slot >= -TRAIL_HOURS && slot <= -1) byHour.set(slot, viewers);
  }
  const trail: number[] = [];
  for (let slot = -TRAIL_HOURS; slot <= -1; slot++) {
    const value = byHour.get(slot);
    if (value === undefined) return null;
    trail.push(value);
  }
  return trail;
}

/**
 * Model B: for each hour, a weighted regression of ln(viewers / pre-release peak) on
 * ln(hour-1 / pre-release peak) over the comparables, then quantiles from the residuals.
 * Returns null when the case does not carry enough for it.
 */
function levelModel(samples: Sample[], targetTrail: number[], p1: number): Map<number, number[]> | null {
  const targetZ = standardiseTrail(targetTrail);
  const targetPeak = Math.max(...targetTrail);
  if (!(targetPeak > 0)) return null;
  const targetB = Math.log(p1 / targetPeak);

  const rows: Array<{ b: number; r: Map<number, number>; w: number }> = [];
  for (const sample of samples) {
    const hour1 = sample.profile[0];
    if (hour1 === null || hour1 < MIN_LEVEL_HOUR1) continue;
    if (sample.trail === null) continue;
    const peak = Math.max(...sample.trail);
    const base = logOf(hour1);
    if (base === null || !(peak > 0)) continue;

    const r = new Map<number, number>();
    for (let hour = FIRST_HOUR; hour <= LAST_HOUR; hour++) {
      const value = logOf(sample.profile[hour - 1]);
      if (value === null) continue;
      r.set(hour, value - Math.log(peak));
    }
    if (r.size < 10) continue;

    const z = standardiseTrail(sample.trail);
    let distance = 0;
    for (let i = 0; i < TRAIL_HOURS; i++) distance += (z[i] - targetZ[i]) ** 2;
    rows.push({ b: base - Math.log(peak), r, w: Math.exp(-distance / (2 * BANDWIDTH)) });
  }
  if (rows.length < MIN_REGRESSION_ROWS) return null;

  // Never extrapolate the regression past the drop the comparables themselves showed: a case whose
  // hour-1 print sits outside their range (a degenerate or truncated print) would otherwise be
  // projected onto a curve far above or below anything observed.
  let minB = Infinity, maxB = -Infinity;
  for (const row of rows) {
    if (row.b < minB) minB = row.b;
    if (row.b > maxB) maxB = row.b;
  }
  const targetDrop = Math.min(maxB, Math.max(minB, targetB));

  // Slope per hour, then smoothed over five hours so one noisy fit cannot swing a row.
  const slope = new Map<number, number>();
  for (let hour = FIRST_HOUR; hour <= LAST_HOUR; hour++) {
    let sw = 0, meanB = 0, meanR = 0;
    const usable = rows.filter((row) => row.r.has(hour));
    if (usable.length < 3) { slope.set(hour, 0); continue; }
    for (const row of usable) { sw += row.w; meanB += row.w * row.b; meanR += row.w * row.r.get(hour)!; }
    meanB /= sw; meanR /= sw;
    let sxy = 0, sxx = 0;
    for (const row of usable) {
      sxy += row.w * (row.b - meanB) * (row.r.get(hour)! - meanR);
      sxx += row.w * (row.b - meanB) ** 2;
    }
    slope.set(hour, sxx > 1e-9 ? sxy / sxx : 0);
  }
  const smoothSlope = new Map<number, number>();
  for (let hour = FIRST_HOUR; hour <= LAST_HOUR; hour++) {
    let sum = 0, n = 0;
    for (let other = hour - 2; other <= hour + 2; other++) {
      const value = slope.get(other);
      if (value !== undefined) { sum += value; n += 1; }
    }
    smoothSlope.set(hour, sum / n);
  }

  const out = new Map<number, number[]>();
  for (let hour = FIRST_HOUR; hour <= LAST_HOUR; hour++) {
    const b = smoothSlope.get(hour) ?? 0;
    let sw = 0, intercept = 0;
    const usable = rows.filter((row) => row.r.has(hour));
    for (const row of usable) { sw += row.w; intercept += row.w * (row.r.get(hour)! - b * row.b); }
    intercept /= sw;
    const predicted = intercept + b * targetDrop;
    const residuals = usable.map((row) => row.r.get(hour)! - (intercept + b * row.b));
    out.set(hour, TAU.map((tau) => targetPeak * Math.exp(predicted + quantile(residuals, tau) * B_SPREAD_SCALE)));
  }
  return out;
}

/** Five viewer quantiles for every forecast hour of one case. */
function forecastCase(caseFile: CaseFile): Array<{ hour: number; q: number[] }> {
  const known = caseFile.known.find((row) => row.hour === 1);
  const p1 = Math.max(1, known && Number.isFinite(known.viewers) ? Number(known.viewers) : 1);
  const dayStart = Date.parse(caseFile.day_start);

  const samples: Sample[] = [];
  for (const comp of caseFile.comparables ?? []) {
    if (!comp || typeof comp !== 'object') continue;
    const sample = toSample(comp, p1);
    if (sample !== null) samples.push(sample);
  }

  const targetTrail = preReleaseTrail(caseFile.history_hourly, dayStart);
  const level = targetTrail !== null && samples.length > 0 ? levelModel(samples, targetTrail, p1) : null;

  const answer: Array<{ hour: number; q: number[] }> = [];
  for (let hour = FIRST_HOUR; hour <= LAST_HOUR; hour++) {
    let quantiles: number[];
    if (samples.length === 0) {
      quantiles = Z.map((z) => p1 * Math.exp(FALLBACK_SHAPE[hour - 2] + z * 0.35));
    } else {
      const shape = shapeModel(samples, hour);
      quantiles = Z.map((z) => p1 * Math.exp(shape.centre + z * shape.spread));
      const fromLevel = level?.get(hour);
      if (fromLevel !== undefined) {
        // Average the two models in log space, then come back to viewers.
        quantiles = quantiles.map((value, index) =>
          Math.exp((1 - BLEND_WEIGHT) * Math.log(value) + BLEND_WEIGHT * Math.log(fromLevel[index])));
      }
    }
    answer.push({ hour, q: tidy(quantiles.map(roundQuantile)) });
  }
  return answer;
}

/** A quantile the answer may carry: finite, above 0, at most 1e9, rounded to two decimals. */
function roundQuantile(value: number): number {
  if (!Number.isFinite(value)) return 1;
  const rounded = Math.round(value * 100) / 100;
  return Math.min(MAX_VIEWERS, Math.max(0.01, rounded));
}

/** Rounding can reorder a pair; the answer format wants the five non-decreasing. */
function tidy(q: number[]): number[] {
  return [...q].sort((a, b) => a - b);
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
