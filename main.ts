// makefun entry for gta6-trophy-rarity-forecast-01.
//
// The input is a case file: the trophy list plus every recorded page of
// Exophase rarity rates. The output is a five-number quantile forecast of
// each trophy's earn rate at the case target.
//
// The model, in short:
//   * identical consecutive pages are one refresh, so a stalled page cannot
//     pretend to be new evidence;
//   * each trophy lives on the logit scale, y = ln(p / (100 - p)), with p
//     clamped to 0.01..99.99;
//   * a recency-weighted, robust line through the distinct refreshes gives a
//     per-day slope when at least three points exist; the slope is pulled
//     toward the tier's median slope and toward zero when the fit is short or
//     noisy, then its whole horizon move is capped at 1.5 logit units;
//   * the median forecast is the last observed y plus 0.75 of that move;
//   * the spread is a robust (MAD, with a trimmed-RMS fallback) scale of
//     one-refresh-ahead changes per square-root day, floored at 0.05, plus a
//     0.10 logit refresh-timing term, and capped at 1.5;
//   * quantiles come from the normal z values on that logit scale, are
//     rounded to four decimals, and are forced to be finite, in range and
//     non-decreasing.
//
// Standard library only. Offline. Reads every *.in under MAKEFUN_INPUT_DIR
// (default /input) and writes <id>.out under MAKEFUN_OUTPUT_DIR (default
// /output).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILE_Z = [-1.6449, -0.6745, 0, 0.6745, 1.6449];
const DAY_MS = 86_400_000;

const RATE_FLOOR = 0.01;
const RATE_CEIL = 99.99;
const VOL_FLOOR = 0.05;
const TIMING_SIGMA = 0.10;
const SIGMA_CAP = 1.5;
const MOVE_CAP = 1.5;
const DAMPING = 0.75;
const RECENCY_HALF_LIFE_DAYS = 0.75;
const DAILY_SLOPE_CAP = 3.0;

interface RawPage {
  at?: string;
  slot?: string;
  players_tracked?: number;
  rates?: Array<number | null>;
}

interface Page {
  /** Milliseconds since the epoch when this distinct page first appeared. */
  time: number;
  rates: Array<number | null>;
  players: number;
}

interface Observation {
  /** Days since the first distinct page. */
  t: number;
  y: number;
}

interface Fit {
  slope: number;
  quality: number;
  n: number;
  /** Approximate standard error of the slope, in logit units per day. */
  se: number;
}

interface TrophyMeta {
  id: number;
  tier?: string;
}

interface CaseFile {
  horizon_minutes?: number;
  trophies?: TrophyMeta[];
  history?: RawPage[];
}

const clamp = (x: number, lo: number, hi: number): number =>
  x < lo ? lo : x > hi ? hi : x;

const isNumber = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v);

/** Percent to logit, with the scorer's clamp. */
const toLogit = (percent: number): number => {
  const p = clamp(percent, RATE_FLOOR, RATE_CEIL);
  return Math.log(p / (100 - p));
};

/** Half away from zero, so both language twins round the same way. */
const round4 = (x: number): number =>
  Math.floor(x * 10_000 + 0.5) / 10_000;

const median = (values: number[]): number => {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
};

const readTime = (page: RawPage, fallback: number): number => {
  const text = page.at ?? page.slot;
  if (typeof text === 'string') {
    const ms = Date.parse(text);
    if (Number.isFinite(ms)) return ms;
  }
  return fallback;
};

/**
 * Keep one page per run of identical consecutive reads. The time of the run
 * is the first read that showed the page: that is when the new information
 * appeared. Repeated reads of the same page add no slope evidence.
 */
const collapsePages = (history: RawPage[]): Page[] => {
  const pages: Page[] = [];
  let previousKey: string | null = null;
  let previousTime = -Infinity;
  history.forEach((raw, index) => {
    const rates = Array.isArray(raw.rates) ? raw.rates : [];
    const players = isNumber(raw.players_tracked) ? raw.players_tracked : -1;
    const key = JSON.stringify(rates) + '|' + String(players);
    if (key === previousKey) return;
    previousKey = key;
    // A clock glitch must not create a negative time step. One second is
    // negligible against the recorder's half-hour cadence.
    const time = Math.max(readTime(raw, index * DAY_MS), previousTime + 1000);
    previousTime = time;
    pages.push({ time, rates, players });
  });
  return pages;
};

/** Point at a line and a robust scale; both are fitted on days and logits. */
const fitRobustLine = (obs: Observation[]): Fit | null => {
  const n = obs.length;
  if (n < 3) return null;

  const latest = obs[n - 1].t;
  let weights = obs.map((o) => {
    const age = Math.max(0, latest - o.t);
    return Math.max(0.02, Math.exp(-age / RECENCY_HALF_LIFE_DAYS));
  });

  let slope = 0;
  let intercept = 0;
  let residuals = new Array<number>(n).fill(0);

  const solve = (): boolean => {
    let sw = 0;
    let st = 0;
    let sy = 0;
    for (let i = 0; i < n; i += 1) {
      sw += weights[i];
      st += weights[i] * obs[i].t;
      sy += weights[i] * obs[i].y;
    }
    if (!(sw > 0)) return false;
    const meanT = st / sw;
    const meanY = sy / sw;
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < n; i += 1) {
      const dt = obs[i].t - meanT;
      sxx += weights[i] * dt * dt;
      sxy += weights[i] * dt * (obs[i].y - meanY);
    }
    if (!(sxx > 1e-12)) return false;
    slope = sxy / sxx;
    intercept = meanY - slope * meanT;
    residuals = obs.map((o) => o.y - (intercept + slope * o.t));
    return true;
  };

  if (!solve()) return null;

  // Two reweighting passes: a single wild page loses its vote, normal noise
  // keeps nearly its full weight.
  for (let pass = 0; pass < 2; pass += 1) {
    const center = median(residuals);
    const scale = 1.4826 * median(residuals.map((r) => Math.abs(r - center)));
    if (!(scale > 1e-9) || n < 4) break;
    const cutoff = 4.685 * scale;
    weights = weights.map((w, i) => {
      const u = residuals[i] / cutoff;
      if (Math.abs(u) >= 1) return w * 0.001;
      const keep = 1 - u * u;
      return w * keep * keep;
    });
    if (!solve()) break;
  }

  let sw = 0;
  let meanY = 0;
  for (let i = 0; i < n; i += 1) {
    sw += weights[i];
    meanY += weights[i] * obs[i].y;
  }
  if (!(sw > 0)) return null;
  meanY /= sw;

  let sst = 0;
  let sse = 0;
  for (let i = 0; i < n; i += 1) {
    const dev = obs[i].y - meanY;
    sst += weights[i] * dev * dev;
    sse += weights[i] * residuals[i] * residuals[i];
  }
  const quality = sst > 1e-12 ? clamp(1 - sse / sst, 0, 1) : 0;

  // A slope measured over a short or ragged line has a large standard error;
  // the caller turns that into extra shrink toward zero.
  const meanT = sw > 0
    ? obs.reduce((sum, o, i) => sum + weights[i] * o.t, 0) / sw
    : 0;
  let centeredSxx = 0;
  for (let i = 0; i < n; i += 1) {
    const dt = obs[i].t - meanT;
    centeredSxx += weights[i] * dt * dt;
  }
  const se = centeredSxx > 1e-12
    ? Math.sqrt(sse / Math.max(1, n - 2) / centeredSxx)
    : 0;

  return { slope, quality, n, se };
};

/**
 * Robust spread of per-square-root-day moves. MAD when it is non-zero; a
 * trimmed RMS when rounded rates make more than half the deviations exactly
 * zero. The trim drops the single largest move, so one bad page cannot set
 * the whole trophy's volatility.
 */
const robustMoveScale = (moves: number[]): number => {
  if (moves.length === 0) return 0;
  const center = median(moves);
  const mad = 1.4826 * median(moves.map((v) => Math.abs(v - center)));
  if (mad > 1e-12) return mad;

  const squares = moves.map((v) => v * v).sort((a, b) => a - b);
  if (squares.length >= 3) squares.pop();
  const rms = Math.sqrt(
    squares.reduce((sum, v) => sum + v, 0) / squares.length,
  );
  return rms;
};

const quantileRow = (yMed: number, sigma: number): number[] => {
  let previous = 0;
  return QUANTILE_Z.map((z) => {
    const y = yMed + z * sigma;
    let p = 100 / (1 + Math.exp(-y));
    if (!Number.isFinite(p)) p = z < 0 ? 0 : 100;
    p = clamp(p, 0, 100);
    let rounded = round4(p);
    if (!Number.isFinite(rounded)) rounded = previous;
    rounded = clamp(rounded, previous, 100);
    previous = rounded;
    return rounded;
  });
};

const forecastCase = (caseFile: CaseFile): { id: number; q: number[] }[] => {
  const trophies = Array.isArray(caseFile.trophies) ? caseFile.trophies : [];
  const pages = collapsePages(
    Array.isArray(caseFile.history) ? caseFile.history : [],
  );
  const horizonDays = Math.max(
    0,
    (isNumber(caseFile.horizon_minutes) ? caseFile.horizon_minutes : 0) / 1440,
  );
  const firstTime = pages.length > 0 ? pages[0].time : 0;

  // One observation series per trophy. Null pages are skipped: the last real
  // observation is the last refresh for that trophy. Pages before the first
  // non-null value simply do not contribute.
  const series = trophies.map(() => [] as Observation[]);
  const lastRates = trophies.map(() => Number.NaN);
  trophies.forEach((_, trophyIndex) => {
    const obs = series[trophyIndex];
    for (const page of pages) {
      const rate = page.rates[trophyIndex];
      if (!isNumber(rate)) continue;
      lastRates[trophyIndex] = rate;
      obs.push({ t: (page.time - firstTime) / DAY_MS, y: toLogit(rate) });
    }
  });

  // First pass: raw fits, and the tier and global slope medians.
  const rawFits = series.map((obs) => fitRobustLine(obs));
  const tierSlopes = new Map<string, number[]>();
  const allSlopes: number[] = [];
  trophies.forEach((trophy, i) => {
    const fit = rawFits[i];
    if (fit === null) return;
    allSlopes.push(fit.slope);
    const tier = typeof trophy.tier === 'string' ? trophy.tier : '';
    const bucket = tierSlopes.get(tier) ?? [];
    bucket.push(fit.slope);
    tierSlopes.set(tier, bucket);
  });
  const globalSlope = allSlopes.length >= 2
    ? median(allSlopes) * (allSlopes.length / (allSlopes.length + 2))
    : 0;

  // Volatility first pass: one-refresh-ahead changes in y per sqrt(day).
  // A trophy's own scale needs at least three moves; with fewer, a single odd
  // page would carry too much weight, so that trophy borrows a pooled scale.
  const trophyMoves = series.map((obs) => {
    const moves: number[] = [];
    for (let i = 1; i < obs.length; i += 1) {
      const dt = obs[i].t - obs[i - 1].t;
      if (!(dt > 1e-9)) continue;
      moves.push((obs[i].y - obs[i - 1].y) / Math.sqrt(dt));
    }
    return moves;
  });
  const ownVol = trophyMoves.map((moves) =>
    moves.length >= 3 ? robustMoveScale(moves) : 0,
  );

  const tierMoves = new Map<string, number[]>();
  const allMoves: number[] = [];
  trophies.forEach((trophy, i) => {
    const tier = typeof trophy.tier === 'string' ? trophy.tier : '';
    const bucket = tierMoves.get(tier) ?? [];
    for (const move of trophyMoves[i]) {
      bucket.push(move);
      allMoves.push(move);
    }
    tierMoves.set(tier, bucket);
  });
  const globalVol = allMoves.length >= 3
    ? Math.max(robustMoveScale(allMoves), VOL_FLOOR)
    : VOL_FLOOR;

  const rows: Array<{ id: number; q: number[] }> = [];
  const emitted = new Set<number>();
  trophies.forEach((trophy, i) => {
    if (!isNumber(trophy.id) || emitted.has(trophy.id)) return;
    emitted.add(trophy.id);

    // Cumulative rarity rates saturate at 0 and 100. The contract asks for
    // literal all-zero and all-hundred rows there, not 0.0118 or 99.9882.
    const lastRaw = lastRates[i];
    if (isNumber(lastRaw) && lastRaw >= 100) {
      rows.push({ id: trophy.id, q: [100, 100, 100, 100, 100] });
      return;
    }
    if (isNumber(lastRaw) && lastRaw <= 0) {
      rows.push({ id: trophy.id, q: [0, 0, 0, 0, 0] });
      return;
    }

    const obs = series[i];
    const fit = rawFits[i];

    // Last observed value, or the nearest earlier page when the final read
    // lacked the trophy. If the trophy never appeared, start at the floor.
    let lastY = toLogit(RATE_FLOOR);
    if (obs.length > 0) lastY = obs[obs.length - 1].y;

    let yMed = lastY;

    if (fit !== null) {
      const tier = typeof trophy.tier === 'string' ? trophy.tier : '';
      const tierList = tierSlopes.get(tier) ?? [];
      // A single fit is too thin to serve as a prior; use the global slope
      // until at least two trophies in the tier agree on a direction.
      let tierSlope = globalSlope;
      if (tierList.length >= 2) {
        const rawTier = median(tierList);
        const strength = tierList.length / (tierList.length + 2);
        tierSlope = rawTier * strength;
      }

      // Confidence grows with the number of refreshes, with fit quality and
      // with the slope's own signal-to-noise ratio. A short, ragged or weak
      // line stays close to the tier prior and to zero.
      const nFactor = clamp((fit.n - 2) / 6, 0, 1);
      const quality = clamp(fit.quality, 0, 1);
      const tRatio = Math.abs(fit.slope) / Math.max(fit.se, 1e-9);
      const signal = tRatio / (tRatio + 1);
      const keep = 0.2 + 0.8 * Math.sqrt(nFactor * quality * signal);
      const tierWeight = clamp(
        0.6 * (1 - nFactor) + 0.25 * (1 - quality),
        0,
        0.8,
      );

      let used = (1 - tierWeight) * fit.slope * keep + tierWeight * tierSlope;
      used = clamp(used, -DAILY_SLOPE_CAP, DAILY_SLOPE_CAP);

      // Smooth damping, then the hard 1.5-logit guard. A very steep recent
      // trend contributes a large but never full linear move.
      const rawMove = used * horizonDays;
      const move = clamp(
        MOVE_CAP * Math.tanh(rawMove / MOVE_CAP),
        -MOVE_CAP,
        MOVE_CAP,
      );
      yMed = lastY + DAMPING * move;
    }

    const tier = typeof trophy.tier === 'string' ? trophy.tier : '';
    const pooledTierMoves = tierMoves.get(tier) ?? [];
    let vol = ownVol[i] > 0 ? ownVol[i] : 0;
    if (!(vol > 0)) {
      vol = pooledTierMoves.length >= 3
        ? robustMoveScale(pooledTierMoves)
        : globalVol;
    }
    vol = Math.max(vol, VOL_FLOOR);

    let sigma = Math.sqrt(vol * vol * horizonDays + TIMING_SIGMA ** 2);
    sigma = clamp(sigma, VOL_FLOOR, SIGMA_CAP);

    rows.push({ id: trophy.id, q: quantileRow(yMed, sigma) });
  });

  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const files = readdirSync(INPUT_DIR)
  .filter((name) => name.endsWith('.in'))
  .sort();

for (const file of files) {
  const caseId = file.slice(0, -'.in'.length);
  const parsed = JSON.parse(
    readFileSync(join(INPUT_DIR, file), 'utf8'),
  ) as CaseFile;
  const rows = forecastCase(parsed);
  const payload = JSON.stringify({ trophies: rows });
  writeFileSync(join(OUTPUT_DIR, `${caseId}.out`), `${payload}\n`);
}
