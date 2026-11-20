// gta6-trophy-rarity-forecast-01
//
// Forecasts every trophy's Exophase earn rate at a case target as five
// quantiles. The model has two rate regimes (a saturating logistic curve and
// an earners/players decomposition), a player-count forecast, a story-chain
// correction, and a deterministic Monte Carlo spread.
//
// Run locally:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const DRAWS = 1500;
const SEED = 0x1a2b3c4d;
const HOUR = 3600000;
const DAY = 86400000;

interface Trophy {
  id?: number;
  name?: string;
  description?: string;
  tier?: string;
}

interface ReadRow {
  at?: string;
  players_tracked?: number;
  trophies_earned?: number;
  rates?: Array<number | null>;
}

interface CaseFile {
  id?: string;
  at?: string;
  target?: string;
  horizon_minutes?: number;
  game?: { release_date?: string };
  trophies?: Trophy[];
  history?: ReadRow[];
}

interface State {
  t: number;
  p: number;
  rates: Array<number | null>;
}

interface Fit {
  k: number;
  t0: number;
}

interface BParams {
  rLast: number;
  eLast: number;
  pLast: number;
  m: number;
}

interface Model {
  kind: string;
  rLast: number;
  ceiling: number;
  fit: Fit | null;
  b: BParams | null;
}

interface Row {
  id: number;
  q: number[];
}

function num(x: unknown, fallback: number): number {
  return typeof x === 'number' && Number.isFinite(x) ? x : fallback;
}

function clamp(x: number, lo: number, hi: number): number {
  if (!Number.isFinite(x)) return lo;
  return Math.min(hi, Math.max(lo, x));
}

function logitPct(pct: number): number {
  const x = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(x / (1 - x));
}

function logitFraction(f: number): number {
  const x = clamp(f, 1e-6, 1 - 1e-6);
  return Math.log(x / (1 - x));
}

function sigmoidPct(l: number): number {
  const x = 1 / (1 + Math.exp(-clamp(l, -60, 60)));
  return clamp(x * 100, 0.01, 99.99);
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function parseMs(s: unknown): number | null {
  if (typeof s !== 'string' || s.length === 0) return null;
  const ms = Date.parse(s);
  return Number.isFinite(ms) ? ms : null;
}

function releaseMs(s: unknown): number | null {
  if (typeof s !== 'string' || s.length === 0) return null;
  const text = /^\d{4}-\d{2}-\d{2}$/.test(s) ? s + 'T00:00:00Z' : s;
  const ms = Date.parse(text);
  return Number.isFinite(ms) ? ms : null;
}

function normalize(s: unknown): string {
  return String(s === undefined || s === null ? '' : s)
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function createRng(seed: number): () => number {
  let s = seed >>> 0;
  if (s === 0) s = 0x6d2b79f5;
  return function next(): number {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

function makeGauss(rng: () => number): () => number {
  let spare: number | null = null;
  return function gauss(): number {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

function poissonSample(lambda: number, rng: () => number, gauss: () => number): number {
  if (!Number.isFinite(lambda) || lambda <= 0) return 0;
  if (lambda < 30) {
    const limit = Math.exp(-lambda);
    let k = 0;
    let prod = 1;
    do {
      k++;
      prod *= rng();
    } while (prod > limit && k < 500);
    return k - 1;
  }
  const z = gauss();
  const v = Math.round(lambda + Math.sqrt(lambda) * z);
  return v > 0 ? v : 0;
}

function sameRates(a: Array<number | null> | null, b: Array<number | null> | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i];
    if (x === null || x === undefined) {
      if (!(y === null || y === undefined)) return false;
    } else if (y === null || y === undefined || x !== y) {
      return false;
    }
  }
  return true;
}

function buildStates(c: CaseFile): { states: State[]; launch: number } {
  const raw = Array.isArray(c.history) ? c.history : [];
  const reads: Array<{ ms: number; p: number; rates: Array<number | null> }> = [];
  for (const h of raw) {
    if (!h || typeof h !== 'object') continue;
    const ms = parseMs(h.at);
    if (ms === null) continue;
    const pRaw = num(h.players_tracked, 0);
    const p = pRaw > 0 ? pRaw : 0;
    const rates = Array.isArray(h.rates) ? h.rates : [];
    reads.push({ ms, p, rates });
  }
  reads.sort((a, b) => a.ms - b.ms);

  let launch = releaseMs(c.game ? c.game.release_date : undefined);
  if (launch === null && reads.length > 0) launch = reads[0].ms;
  if (launch === null) {
    const at = parseMs(c.at);
    const target = parseMs(c.target);
    launch = at !== null ? at : target !== null ? target : 0;
  }

  const states: State[] = [];
  let previous: Array<number | null> | null = null;
  for (const r of reads) {
    const effective: Array<number | null> = [];
    const n = Math.max(r.rates.length, previous ? previous.length : 0);
    for (let i = 0; i < n; i++) {
      const v = r.rates[i];
      if (typeof v === 'number' && Number.isFinite(v)) effective.push(v);
      else effective.push(previous ? (previous[i] ?? null) : null);
    }
    const last = states.length > 0 ? states[states.length - 1] : null;
    if (last && last.p === r.p && sameRates(last.rates, effective)) {
      previous = last.rates;
      continue;
    }
    states.push({ t: (r.ms - launch) / HOUR, p: r.p, rates: effective });
    previous = effective;
  }
  return { states, launch };
}

function rateOf(s: State, i: number): number | null {
  const v = s.rates[i];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function playerModel(states: State[], targetT: number) {
  const n = states.length;
  const last = states[n - 1];
  const lastP = last.p > 0 ? last.p : 1;
  let g = 0;
  let interval = 0;
  const multi = n >= 2;
  if (multi) {
    const prev = states[n - 2];
    interval = last.t - prev.t;
    if (interval >= 0.5 && prev.p > 0 && last.p > 0) {
      g = Math.log(last.p / prev.p) / interval;
    }
    if (!Number.isFinite(g)) g = 0;
    g = clamp(g, 0, 0.15);
  }
  const gEff = multi ? (g * 36) / (interval + 36) : 0;

  const multAt = (tt: number): number => {
    if (multi) {
      const dt = tt - last.t;
      return clamp(Math.exp(gEff * dt), 1.25, 10);
    }
    const dt = Math.max(0, tt - last.t);
    return clamp(3 * Math.sqrt(dt / 108), 1.2, 8);
  };

  const nominal = multAt(targetT);
  const dtDays = Math.max(0, targetT - last.t) / 24;
  const sigma = 0.35 + 0.08 * dtDays;
  return { lastP, lastT: last.t, g, gEff, multi, interval, multAt, nominal, sigma, targetT };
}

function ceilingFor(t: Trophy): number {
  if (normalize(t.tier) === 'platinum') return 0.006;
  const d = normalize(t.description);
  const both = d + ' ' + normalize(t.name);
  if (d.startsWith('complete chapter 1')) return 0.55;
  if (d.startsWith('complete chapter 2')) return 0.32;
  if (d.startsWith('complete chapter 3')) return 0.16;
  if (d.startsWith('complete chapter 4')) return 0.09;
  if (/^complete the final mission\.?$/.test(d)) return 0.08;
  if (d.includes('complete breakfast run')) return 0.30;
  if (d.includes('beat cal in a fishing challenge')) return 0.40;
  if (d.includes('complete an arcade game')) return 0.20;
  if (both.includes('kolt') || both.includes('bowling')) return 0.16;
  if (d.includes('discover 100 vehicles')) return 0.15;
  if (d.includes('what the hell just happened')) return 0.08;
  if (d.includes('lose a 4 star wanted level')) return 0.07;
  if (d.includes('make an unscheduled departure')) return 0.06;
  if (d.includes('kinda hot in these rhinos')) return 0.06;
  if (d.includes('enjoy a cocktail with a view')) return 0.05;
  if (d.includes('attain 100% completion')) return 0.012;
  return 0.05;
}

function singleFit(r: number, t1: number, cFit: number): Fit {
  const c = Math.max(cFit, 1e-6);
  let y = r / 100;
  if (!Number.isFinite(y) || y <= 0) y = 1e-5;
  if (y >= c) y = c * 0.98;
  const f0 = clamp(y / c, 1e-4, 0.98);
  const y96 = y + 0.75 * (c - y);
  const f96 = clamp(y96 / c, Math.min(0.999, f0 + 1e-6), 0.999);
  let k = (logitFraction(f96) - logitFraction(f0)) / 96;
  if (!Number.isFinite(k) || k <= 0) k = 0.01;
  k = clamp(k, 0.001, 1.5);
  const t0 = t1 - logitFraction(f0) / k;
  return { k, t0 };
}

function fitRegimeA(states: State[], trophyIndex: number, ceiling: number): Fit | null {
  const usable: Array<{ t: number; r: number }> = [];
  for (const s of states) {
    const r = rateOf(s, trophyIndex);
    if (r !== null && r >= 0.02 && s.t >= 4) usable.push({ t: s.t, r });
  }

  if (usable.length === 0) {
    for (let j = states.length - 1; j >= 0; j--) {
      const r = rateOf(states[j], trophyIndex);
      if (r !== null) {
        usable.push({ t: states[j].t, r });
        break;
      }
    }
  }
  if (usable.length === 0) return null;

  let maxR = 0.02;
  for (const u of usable) if (u.r > maxR) maxR = u.r;
  const cFit = Math.max(ceiling, (maxR / 100) * 1.01, 1e-6);

  if (usable.length === 1) {
    return singleFit(usable[0].r, usable[0].t, cFit);
  }

  const weightFor = (age: number): number => {
    if (age === 0) return 1;
    if (age === 1) return 0.6;
    if (age === 2) return 0.35;
    if (age === 3) return 0.2;
    return 0.2 * Math.pow(0.5, age - 3);
  };

  const points: Array<{ x: number; z: number; w: number }> = [];
  let sw = 0;
  let sx = 0;
  let sz = 0;
  for (let j = 0; j < usable.length; j++) {
    const age = usable.length - 1 - j;
    const w = weightFor(age);
    const u = usable[j];
    let f = u.r / 100 / cFit;
    f = clamp(f, 0.0005, 0.98);
    const z = Math.log(f / (1 - f));
    points.push({ x: u.t, z, w });
    sw += w;
    sx += w * u.t;
    sz += w * z;
  }

  const xbar = sx / sw;
  const zbar = sz / sw;
  let sxx = 0;
  let sxz = 0;
  for (const p of points) {
    sxx += p.w * (p.x - xbar) * (p.x - xbar);
    sxz += p.w * (p.x - xbar) * (p.z - zbar);
  }

  let k = sxx > 1e-9 ? sxz / sxx : 0;
  if (!Number.isFinite(k) || k <= 0) {
    const u = usable[usable.length - 1];
    return singleFit(u.r, u.t, cFit);
  }
  k = clamp(k, 0.001, 1.5);
  const t0 = xbar - zbar / k;
  return { k, t0 };
}

function bParams(states: State[], i: number, rLast: number): BParams {
  const n = states.length;
  const last = states[n - 1];
  const pLast = last.p > 0 ? last.p : 1;
  const eLast = (rLast * pLast) / 100;
  let m = (0.5 * rLast) / 100;

  if (n >= 2) {
    const prev = states[n - 2];
    const rPrev = rateOf(prev, i);
    const pPrev = prev.p;
    const growth = last.p - pPrev;
    const interval = last.t - prev.t;
    if (rPrev !== null && growth >= 100 && interval >= 6 && pPrev > 0) {
      const ePrev = (rPrev * pPrev) / 100;
      m = (eLast - ePrev) / growth;
    }
  }
  m = clamp(m, 0, 0.9);
  return { rLast, eLast, pLast, m };
}

function prepareModel(states: State[], i: number, trophy: Trophy, prologue: boolean): Model {
  const last = states[states.length - 1];
  const raw = rateOf(last, i);
  const rLast = raw === null ? 0.01 : clamp(raw, 0.01, 99.99);
  if (prologue) return { kind: 'prologue', rLast: 100, ceiling: 1, fit: null, b: null };
  const ceiling = ceilingFor(trophy);
  const fit = rLast >= 0.25 ? fitRegimeA(states, i, ceiling) : null;
  const b = bParams(states, i, rLast);
  return { kind: 'normal', rLast, ceiling, fit, b };
}

function regimeBRate(
  b: BParams | null,
  pTarget: number,
  gauss: () => number,
  rng: () => number,
): number {
  if (!b) return 0.01;
  const m = b.m > 0 ? clamp(b.m * Math.exp(0.45 * gauss()), 0, 0.95) : 0;
  let earners = b.eLast + m * (pTarget - b.pLast);
  if (!Number.isFinite(earners) || earners <= 0) return 0.01;
  const noisy = poissonSample(earners, rng, gauss);
  if (noisy <= 0) return 0.01;
  let rate = pTarget > 0 ? (100 * noisy) / pTarget : 0.01;
  if (!Number.isFinite(rate) || rate < 0.005) rate = 0.01;
  return clamp(rate, 0.01, 99.99);
}

function simulate(
  model: Model,
  pTarget: number,
  tEff: number,
  gauss: () => number,
  rng: () => number,
): number {
  if (model.kind === 'prologue') return 100;
  const rLast = model.rLast;
  let combined: number;

  if (rLast >= 0.25 && model.fit) {
    const cDraw = clamp(model.ceiling * Math.exp(0.35 * gauss()), 1e-5, 0.999);
    const kDraw = clamp(model.fit.k * Math.exp(0.30 * gauss()), 1e-4, 2);
    let rA = (100 * cDraw) / (1 + Math.exp(-clamp(kDraw * (tEff - model.fit.t0), -60, 60)));
    const cap = Math.max(98 * cDraw, rLast);
    if (rA > cap) rA = cap;
    rA = clamp(rA, 0.01, 99.99);
    const rB = regimeBRate(model.b, pTarget, gauss, rng);
    combined = 0.65 * logitPct(rA) + 0.35 * logitPct(rB);
  } else {
    const rB = regimeBRate(model.b, pTarget, gauss, rng);
    combined = logitPct(rB);
  }

  if (rLast < 0.06) combined = 0.8 * combined + 0.2 * logitPct(rLast);

  let out = sigmoidPct(combined);
  if (!Number.isFinite(out) || out < 0.005) out = 0.01;
  out = Math.round(out * 100) / 100;
  return clamp(out, 0.01, 99.99);
}

function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0.01;
  if (sorted.length === 1) return sorted[0];
  const pos = p * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  const f = pos - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * f;
}

function makeRow(id: number, draws: number[]): Row {
  const sorted = draws.slice().sort((a, b) => a - b);
  const q: number[] = [];
  for (const p of LEVELS) {
    const raw = quantile(sorted, p);
    const rounded = round4(clamp(raw, 0.01, 99.99));
    const floored = Number.isFinite(rounded) ? Math.max(0.01, rounded) : 0.01;
    q.push(floored);
  }
  // Re-check after rounding so every row is finite, floored and non-decreasing.
  for (let i = 0; i < q.length; i++) {
    q[i] = Math.max(0.01, q[i]);
    if (i > 0 && q[i] < q[i - 1]) q[i] = q[i - 1];
  }
  return { id, q };
}

function baselineRow(id: number, rLast: number, horizonDays: number): Row {
  const l = logitPct(Number.isFinite(rLast) ? rLast : 0.01);
  const spread = 0.5 * Math.sqrt(Math.max(0, Number.isFinite(horizonDays) ? horizonDays : 0));
  const q = Z.map((z) => round4(clamp(sigmoidPct(l + z * spread), 0.01, 99.99)));
  for (let i = 1; i < q.length; i++) {
    if (q[i] < q[i - 1]) q[i] = q[i - 1];
  }
  return { id, q };
}

function medianLogit(draws: number[]): number {
  const a = draws.map(logitPct).sort((x, y) => x - y);
  if (a.length === 0) return 0;
  const n = a.length;
  if (n % 2 === 1) return a[(n - 1) / 2];
  return (a[n / 2 - 1] + a[n / 2]) / 2;
}

function shiftDraws(draws: number[], delta: number): void {
  if (!Number.isFinite(delta) || delta === 0) return;
  for (let i = 0; i < draws.length; i++) {
    draws[i] = sigmoidPct(logitPct(draws[i]) + delta);
  }
}

function isotonicNonIncreasing(values: number[]): number[] {
  const blocks: Array<{ sum: number; count: number }> = [];
  for (const v of values) {
    blocks.push({ sum: v, count: 1 });
    while (blocks.length >= 2) {
      const a = blocks[blocks.length - 2];
      const b = blocks[blocks.length - 1];
      if (a.sum / a.count >= b.sum / b.count) break;
      blocks.pop();
      a.sum += b.sum;
      a.count += b.count;
    }
  }
  const out: number[] = [];
  for (const b of blocks) {
    const mean = b.sum / b.count;
    for (let i = 0; i < b.count; i++) out.push(mean);
  }
  return out;
}

function applyStoryCorrections(
  trophies: Trophy[],
  draws: number[][],
  broken: boolean[],
  prologueIndex: number,
): void {
  const locate = (pred: (t: Trophy) => boolean, fallbackId: number): number => {
    for (let i = 0; i < trophies.length; i++) {
      if (!broken[i] && pred(trophies[i])) return i;
    }
    for (let i = 0; i < trophies.length; i++) {
      if (!broken[i] && trophies[i].id === fallbackId) return i;
    }
    return -1;
  };

  const desc = (t: Trophy): string => normalize(t.description);
  const ch1 = locate((t) => desc(t).startsWith('complete chapter 1'), 3);
  const ch2 = locate((t) => desc(t).startsWith('complete chapter 2'), 4);
  const ch3 = locate((t) => desc(t).startsWith('complete chapter 3'), 5);
  const ch4 = locate((t) => desc(t).startsWith('complete chapter 4'), 6);
  const finalMission = locate((t) => /^complete the final mission\.?$/.test(desc(t)), 7);
  const breakfast = locate((t) => desc(t).includes('complete breakfast run'), 8);
  const platinum = locate((t) => normalize(t.tier) === 'platinum', 1);

  const chain = [ch1, ch2, ch3, ch4, finalMission];
  const unique = new Set(chain);
  if (chain.every((i) => i >= 0) && unique.size === 5) {
    const medians = chain.map((i) => medianLogit(draws[i]));
    const corrected = isotonicNonIncreasing(medians);
    for (let j = 0; j < chain.length; j++) {
      shiftDraws(draws[chain[j]], corrected[j] - medians[j]);
    }
  }

  if (ch1 >= 0 && ch2 >= 0 && breakfast >= 0 && new Set([ch1, ch2, breakfast]).size === 3) {
    const m1 = medianLogit(draws[ch1]);
    const m2 = medianLogit(draws[ch2]);
    const mb = medianLogit(draws[breakfast]);
    const lo = Math.min(m1, m2);
    const hi = Math.max(m1, m2);
    const wanted = clamp(mb, lo, hi);
    shiftDraws(draws[breakfast], wanted - mb);
  }

  if (platinum >= 0) {
    let lowest = Infinity;
    for (let i = 0; i < trophies.length; i++) {
      if (i === prologueIndex || broken[i] || draws[i].length !== DRAWS) continue;
      const m = medianLogit(draws[i]);
      if (m < lowest) lowest = m;
    }
    if (Number.isFinite(lowest)) {
      const mp = medianLogit(draws[platinum]);
      if (mp > lowest) shiftDraws(draws[platinum], lowest - mp);
    }
  }
}

function trophyId(t: Trophy, i: number): number {
  return typeof t.id === 'number' && Number.isFinite(t.id) ? t.id : i + 1;
}

function caseHorizonDays(c: CaseFile): number {
  const hm = num(c.horizon_minutes, NaN);
  if (Number.isFinite(hm) && hm >= 0) return hm / 1440;
  const at = parseMs(c.at);
  const target = parseMs(c.target);
  if (at !== null && target !== null) return Math.max(0, target - at) / DAY;
  return 1.5 / 24;
}

function lastReadRates(c: CaseFile, n: number): number[] {
  const history = Array.isArray(c.history) ? c.history : [];
  const out: number[] = new Array(n).fill(0.01);
  for (let i = 0; i < n; i++) {
    for (let j = history.length - 1; j >= 0; j--) {
      const h = history[j];
      if (!h || !Array.isArray(h.rates)) continue;
      const v = h.rates[i];
      if (typeof v === 'number' && Number.isFinite(v)) {
        out[i] = v;
        break;
      }
    }
  }
  return out;
}

function processCase(c: CaseFile): { trophies: Row[] } {
  const trophies = Array.isArray(c.trophies)
    ? c.trophies.filter((t) => t && typeof t === 'object')
    : [];
  const horizonDays = caseHorizonDays(c);
  const lastRates = lastReadRates(c, trophies.length);
  const baselineAll = (): Row[] =>
    trophies.map((t, i) => baselineRow(trophyId(t, i), lastRates[i] ?? 0.01, horizonDays));

  try {
    const built = buildStates(c);
    const states = built.states;
    if (states.length === 0) return { trophies: baselineAll() };

    const targetMs = parseMs(c.target);
    if (targetMs === null) throw new Error('target is not a time');
    const targetT = (targetMs - built.launch) / HOUR;
    const pm = playerModel(states, targetT);

    let prologueIndex = -1;
    for (let i = 0; i < trophies.length; i++) {
      if (trophies[i].id === 2) {
        prologueIndex = i;
        break;
      }
    }
    if (prologueIndex < 0) {
      for (let i = 0; i < trophies.length; i++) {
        const r = rateOf(states[states.length - 1], i);
        if (r !== null && r >= 99.995) {
          prologueIndex = i;
          break;
        }
      }
    }

    const models: Array<Model | null> = [];
    const broken: boolean[] = [];
    for (let i = 0; i < trophies.length; i++) {
      try {
        models.push(prepareModel(states, i, trophies[i], i === prologueIndex));
        broken.push(false);
      } catch {
        models.push(null);
        broken.push(true);
      }
    }

    const draws: number[][] = trophies.map(() => []);
    const rng = createRng(SEED);
    const gauss = makeGauss(rng);

    for (let d = 0; d < DRAWS; d++) {
      const offset = 24 * rng();
      const tEff = targetT - offset;
      const pTarget = Math.max(
        1,
        pm.lastP * pm.multAt(tEff) * Math.exp(pm.sigma * gauss()),
      );
      for (let i = 0; i < trophies.length; i++) {
        const model = models[i];
        if (!model || broken[i]) {
          draws[i].push(NaN);
          continue;
        }
        try {
          draws[i].push(simulate(model, pTarget, tEff, gauss, rng));
        } catch {
          broken[i] = true;
          draws[i].push(NaN);
        }
      }
    }

    for (let i = 0; i < trophies.length; i++) {
      if (broken[i]) continue;
      for (const v of draws[i]) {
        if (!Number.isFinite(v)) {
          broken[i] = true;
          break;
        }
      }
    }

    applyStoryCorrections(trophies, draws, broken, prologueIndex);

    const rows: Row[] = trophies.map((t, i) => {
      const id = trophyId(t, i);
      if (i === prologueIndex) return { id, q: [100, 100, 100, 100, 100] };
      if (broken[i] || draws[i].length !== DRAWS) {
        return baselineRow(id, lastRates[i] ?? 0.01, horizonDays);
      }
      return makeRow(id, draws[i]);
    });
    return { trophies: rows };
  } catch {
    return { trophies: baselineAll() };
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });

let files: string[] = [];
try {
  files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
} catch {
  files = [];
}

for (const file of files) {
  const id = file.slice(0, -'.in'.length);
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8'));
  } catch {
    parsed = null;
  }

  let payload: { trophies: Row[] };
  if (parsed && typeof parsed === 'object') {
    payload = processCase(parsed as CaseFile);
  } else {
    payload = { trophies: [] };
  }

  try {
    writeFileSync(join(OUTPUT_DIR, id + '.out'), JSON.stringify(payload) + '\n', 'utf8');
  } catch {
    // Nothing to write to; the next case may still succeed.
  }
}
