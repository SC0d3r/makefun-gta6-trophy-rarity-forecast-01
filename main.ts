import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const OBS_WEIGHT = 0.42;
const PRIOR_WEIGHT = 0.58;
const DECAY = 0.6;
const ANCHOR_WEIGHT = 0.35;
const RATIO_PRIOR = 1.8;
const ABS_PRIOR = 1.0;
const SIGMA_STORY = 1.10;
const SIGMA_SIDE = 1.20;

const IN_DIR = process.env.MAKEFUN_INPUT_DIR || "/input";
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR || "/output";

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function logitPct(p: number): number {
  const q = clamp(p, 0.01, 99.99) / 100;
  return Math.log(q / (1 - q));
}

function logistic(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function parseTime(s: unknown): number {
  const t = typeof s === "string" ? Date.parse(s) : NaN;
  return Number.isFinite(t) ? t : NaN;
}

function hoursBetween(a: string, b: string): number {
  const ta = parseTime(a);
  const tb = parseTime(b);
  if (!Number.isFinite(ta) || !Number.isFinite(tb)) return 0.25;
  return (ta - tb) / 3600000;
}

function cleanRate(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? clamp(v, 0, 100) : NaN;
}

function xOf(hours: number): number {
  return Math.log(1 + Math.max(0.25, hours));
}

type Trophy = {
  id: number;
  name?: string;
  description?: string;
  tier?: string;
  points?: number;
  secret?: boolean;
};

type Read = {
  slot?: string;
  at?: string;
  players_tracked?: number;
  trophies_earned?: number;
  rates?: Array<number | null>;
};

type Case = {
  id: string;
  at: string;
  target: string;
  horizon_minutes?: number;
  game?: { release_date?: string };
  trophies: Trophy[];
  history: Read[];
};

type State = {
  at: string;
  players: number;
  rates: number[];
};

type Row = {
  id: number;
  y: number;
  sigma: number;
  prologue: boolean;
  lastRate: number;
  cls: string;
  desc: string;
};

function classify(t: Trophy, lastRate: number): string {
  const desc = String(t.description || "");
  const tier = String(t.tier || "").toLowerCase();
  if (desc.includes("Gone Fishing")) return "PROLOGUE";
  if (tier === "bronze" && lastRate > 99.5) return "PROLOGUE";
  if (tier === "platinum") return "PLATINUM";
  if (/^Complete Chapter [0-9]/.test(desc)) return "STORY";
  if (/^Complete the final mission\./.test(desc)) return "STORY";
  if (
    desc === "Complete Breakfast Run" ||
    desc === "Complete The Cartel Heist" ||
    desc === "Complete Child's Play" ||
    desc === "Complete Headliners" ||
    desc === "Complete VCIA Takedown" ||
    desc === "Complete Operation Savage Glory"
  ) {
    return "STORY";
  }
  if (desc.includes("Attain 100% Completion")) return "COMPLETION";
  if (/^Complete the final mission for /.test(desc)) return "MISSION";
  return "SIDE";
}

function priorSlope(cls: string, lastRate: number): number {
  if (cls === "STORY") return 1.25;
  if (cls === "MISSION") return 1.0;
  if (cls === "COMPLETION") return 0.85;
  if (cls === "PLATINUM") return 0.9;
  if (cls === "SIDE") return 0.5 + 0.3 * Math.min(1, Math.max(0, lastRate) / 0.5);
  return 0;
}

// Weighted least squares slope of y on x.
function wlsSlope(xs: number[], ys: number[], ws: number[]): number {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (!(sw > 0)) return 0;
  const xb = sx / sw;
  const yb = sy / sw;
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - xb;
    num += ws[i] * dx * (ys[i] - yb);
    den += ws[i] * dx * dx;
  }
  if (!(den > 1e-12)) return 0;
  const s = num / den;
  return Number.isFinite(s) ? s : 0;
}

function observedSlope(states: State[], ti: number, lastX: number): number {
  if (states.length < 2) return 0;
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  for (const st of states) {
    const r = st.rates[ti];
    const rate = Number.isFinite(r) ? r : 0.01;
    const x = stateX(st);
    xs.push(x);
    ys.push(logitPct(rate));
    ws.push(Math.exp(-(lastX - x) / DECAY));
  }
  let slope = wlsSlope(xs, ys, ws);
  // Second pass: clip residuals to +-1.0 and refit.
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw > 0) {
    const xb = sx / sw;
    const yb = sy / sw;
    const adj: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      const fit = yb + slope * (xs[i] - xb);
      const resid = clamp(ys[i] - fit, -1.0, 1.0);
      adj.push(fit + resid);
    }
    slope = wlsSlope(xs, adj, ws);
  }
  return clamp(slope, -1.2, 2.0);
}

function stateX(st: State): number {
  return xOf(hoursBetween(st.at, EPOCH));
}

let EPOCH = "1970-01-01T00:00:00Z";

function pavaNonIncreasing(vals: number[]): number[] {
  // Pool adjacent violators for a non-increasing sequence.
  type Block = { sum: number; count: number };
  const blocks: Block[] = [];
  for (const v of vals) {
    blocks.push({ sum: v, count: 1 });
    while (blocks.length >= 2) {
      const a = blocks[blocks.length - 2];
      const b = blocks[blocks.length - 1];
      if (a.sum / a.count >= b.sum / b.count) break;
      blocks.pop();
      blocks.pop();
      blocks.push({ sum: a.sum + b.sum, count: a.count + b.count });
    }
  }
  const out: number[] = [];
  for (const b of blocks) {
    const m = b.sum / b.count;
    for (let i = 0; i < b.count; i++) out.push(m);
  }
  return out;
}

function processCase(c: Case): Row[] {
  const atMs = parseTime(c.at);
  const relMs = parseTime(String(c.game?.release_date || "") + "T00:00:00Z");
  const epochMs = Number.isFinite(relMs)
    ? relMs
    : Number.isFinite(atMs)
      ? Math.floor(atMs / 86400000) * 86400000
      : 0;
  EPOCH = new Date(epochMs).toISOString();
  const atFallback = Number.isFinite(atMs) ? c.at : EPOCH;
  const target = Number.isFinite(parseTime(c.target)) ? c.target : atFallback;
  const history = Array.isArray(c.history) ? c.history : [];
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const n = trophies.length;

  // a. Collapse to distinct page states, carrying null rates forward.
  const states: State[] = [];
  let prevSig = "";
  let carry: number[] = new Array(n).fill(0.01);
  for (const h of history) {
    const ratesIn = Array.isArray(h.rates) ? h.rates : [];
    const raw: Array<number | null> = [];
    for (let i = 0; i < n; i++) {
      const v = cleanRate(ratesIn[i]);
      raw.push(Number.isFinite(v) ? v : null);
    }
    const sig = JSON.stringify([h.players_tracked, raw]);
    if (sig === prevSig) continue;
    prevSig = sig;
    const carried: number[] = [];
    for (let i = 0; i < n; i++) {
      const v = raw[i];
      if (v !== null) {
        carry[i] = v;
        carried.push(v);
      } else {
        carried.push(carry[i]);
      }
    }
    const at = typeof h.at === "string" && Number.isFinite(parseTime(h.at)) ? h.at : atFallback;
    states.push({
      at,
      players: typeof h.players_tracked === "number" && Number.isFinite(h.players_tracked) ? h.players_tracked : 0,
      rates: carried,
    });
  }

  const lastState = states.length > 0 ? states[states.length - 1] : null;
  const lastRates: number[] = lastState ? lastState.rates.slice() : new Array(n).fill(0.01);
  const lastX = lastState ? stateX(lastState) : xOf(0.25);
  const targetX = xOf(hoursBetween(target, EPOCH));

  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    const t = trophies[i];
    const lr = cleanRate(lastRates[i]);
    const lastRate = clamp(Number.isFinite(lr) ? lr : 0.01, 0.01, 99.99);
    const cls = classify(t, lastRate);
    if (cls === "PROLOGUE") {
      rows.push({ id: t.id, y: logitPct(99.99), sigma: 0, prologue: true, lastRate, cls, desc: String(t.description || "") });
      continue;
    }
    const obs = observedSlope(states, i, lastX);
    const prior = priorSlope(cls, lastRate);
    const s = clamp(PRIOR_WEIGHT * prior + OBS_WEIGHT * obs, 0.1, 1.7);
    const y = logitPct(lastRate) + s * (targetX - lastX);
    const sigma = cls === "SIDE" ? SIGMA_SIDE : SIGMA_STORY;
    rows.push({ id: t.id, y, sigma, prologue: false, lastRate, cls, desc: String(t.description || "") });
  }

  // h. Cross-game sum anchor.
  const nonPrologue = rows.filter((r) => !r.prologue);
  let sPred = 0;
  let sObs = 0;
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    if (r.prologue) continue;
    sPred += logistic(r.y);
    const so = cleanRate(lastRates[i]);
    sObs += clamp(Number.isFinite(so) ? so : 0.01, 0.01, 99.99) / 100;
  }
  if (sPred > 0.02 && nonPrologue.length > 0) {
    const sPrior = Math.sqrt(Math.max(0.02, sObs) * RATIO_PRIOR * ABS_PRIOR);
    let sTarget = Math.exp(0.65 * Math.log(sPred) + ANCHOR_WEIGHT * Math.log(sPrior));
    sTarget = clamp(sTarget, 0.15, 2.2);
    const sumAt = (k: number): number => {
      let s = 0;
      for (const r of nonPrologue) s += logistic(r.y + k);
      return s;
    };
    let lo = -2;
    let hi = 2;
    for (let step = 0; step < 40; step++) {
      const mid = (lo + hi) / 2;
      if (sumAt(mid) < sTarget) lo = mid;
      else hi = mid;
    }
    const k = (lo + hi) / 2;
    for (const r of nonPrologue) r.y += k;
  }

  // i. Monotonicity: story chain in id order, then platinum below everything.
  const chain = rows
    .filter(
      (r) =>
        r.cls === "STORY" &&
        (/^Complete Chapter [1-5]\./.test(r.desc) || r.desc === "Complete the final mission."),
    )
    .sort((a, b) => a.id - b.id);
  if (chain.length > 1) {
    const fixed = pavaNonIncreasing(chain.map((r) => r.y));
    for (let i = 0; i < chain.length; i++) chain[i].y = fixed[i];
  }
  const plat = rows.filter((r) => r.cls === "PLATINUM");
  if (plat.length > 0) {
    let minOther = Infinity;
    for (const r of rows) if (r.cls !== "PLATINUM") minOther = Math.min(minOther, r.y);
    if (Number.isFinite(minOther)) {
      for (const r of plat) r.y = Math.min(r.y, minOther);
    }
  }

  // j. Spread, including staleness of the last kept page.
  let stale = 0;
  const atForStale = parseTime(atFallback);
  const lastReadMs = lastState ? parseTime(lastState.at) : NaN;
  if (Number.isFinite(atForStale) && Number.isFinite(lastReadMs)) {
    const age = (atForStale - lastReadMs) / 3600000;
    stale = Math.min(0.3, Math.floor(Math.max(0, age - 2) / 6) * 0.05);
  }
  for (const r of rows) {
    if (r.prologue) {
      r.sigma = 0;
      continue;
    }
    let s = r.sigma + stale;
    if (r.lastRate < 0.1) s += 0.08;
    r.sigma = Math.min(1.5, s);
  }

  return rows;
}

function quantiles(row: Row): number[] {
  if (row.prologue) return [99.99, 99.99, 99.99, 99.99, 99.99];
  const out: number[] = [];
  let run = 0.01;
  for (const z of Z) {
    const p = 100 * logistic(row.y + z * row.sigma);
    const q = clamp(Math.round(clamp(p, 0.01, 99.99) * 10000) / 10000, 0.01, 99.99);
    run = Math.max(run, q);
    out.push(run);
  }
  return out;
}

function main(): void {
  const files = readdirSync(IN_DIR)
    .filter((f) => f.endsWith(".in"))
    .sort();
  mkdirSync(OUT_DIR, { recursive: true });
  for (const f of files) {
    const raw = readFileSync(join(IN_DIR, f), "utf8");
    const c = JSON.parse(raw) as Case;
    const rows = processCase(c);
    const trophies = rows.map((r) => ({ id: r.id, q: quantiles(r) }));
    const out = { trophies };
    const id = String(c.id || f.replace(/\.in$/, ""));
    writeFileSync(join(OUT_DIR, id + ".out"), JSON.stringify(out));
  }
}

main();
