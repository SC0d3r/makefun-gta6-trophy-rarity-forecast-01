#!/usr/bin/env python3
"""makefun entry for gta6-trophy-rarity-forecast-01, Python twin of main.ts.

The case file holds a trophy list and a history of Exophase rarity reads.
This program collapses repeated reads into distinct refreshes, fits a robust
recent slope per trophy on the logit scale, extrapolates a damped median, and
turns a robust residual volatility into five quantiles.

It reads every *.in under MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out under MAKEFUN_OUTPUT_DIR (default /output). Standard library only,
offline, no network.
"""

import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILE_Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]
DAY_MS = 86400000.0

RATE_FLOOR = 0.01
RATE_CEIL = 99.99
VOL_FLOOR = 0.05
TIMING_SIGMA = 0.10
SIGMA_CAP = 1.5
MOVE_CAP = 1.5
DAMPING = 0.75
RECENCY_HALF_LIFE_DAYS = 0.75
DAILY_SLOPE_CAP = 3.0


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def to_logit(percent):
    p = clamp(percent, RATE_FLOOR, RATE_CEIL)
    return math.log(p / (100.0 - p))


def round4(value):
    # Half away from zero, the same rule as the TypeScript twin.
    return math.floor(value * 10000.0 + 0.5) / 10000.0


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def read_time(page, fallback):
    text = page.get("at")
    if not isinstance(text, str):
        text = page.get("slot")
    if isinstance(text, str):
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000.0
        except ValueError:
            pass
    return fallback


def collapse_pages(history):
    """Keep one page per run of identical consecutive reads.

    The page's time is the first read that showed it: repeated reads of the
    same rates and player count are not new evidence.
    """
    pages = []
    previous_rates = None
    previous_players = None
    previous_time = float("-inf")
    for index, raw in enumerate(history):
        if not isinstance(raw, dict):
            continue
        rates = raw.get("rates")
        if not isinstance(rates, list):
            rates = []
        players = raw.get("players_tracked")
        if not is_number(players):
            players = -1
        if rates == previous_rates and players == previous_players:
            continue
        previous_rates = rates
        previous_players = players
        time = max(read_time(raw, index * DAY_MS), previous_time + 1000.0)
        previous_time = time
        pages.append({"time": time, "rates": rates, "players": players})
    return pages


def fit_robust_line(obs):
    """Recency-weighted robust line through (day, logit) observations."""
    n = len(obs)
    if n < 3:
        return None

    xs = [point[0] for point in obs]
    ys = [point[1] for point in obs]
    latest = xs[-1]
    weights = [max(0.02, math.exp(-max(0.0, latest - x) / RECENCY_HALF_LIFE_DAYS)) for x in xs]

    slope = 0.0
    intercept = 0.0
    residuals = [0.0] * n

    def solve():
        nonlocal slope, intercept, residuals
        total_w = sum(weights)
        if not total_w > 0:
            return False
        mean_t = sum(weights[i] * xs[i] for i in range(n)) / total_w
        mean_y = sum(weights[i] * ys[i] for i in range(n)) / total_w
        sxx = 0.0
        sxy = 0.0
        for i in range(n):
            dt = xs[i] - mean_t
            sxx += weights[i] * dt * dt
            sxy += weights[i] * dt * (ys[i] - mean_y)
        if not sxx > 1e-12:
            return False
        slope = sxy / sxx
        intercept = mean_y - slope * mean_t
        residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
        return True

    if not solve():
        return None

    # Two passes of bisquare reweighting: one wild page loses its vote.
    for _ in range(2):
        center = median(residuals)
        scale = 1.4826 * median([abs(r - center) for r in residuals])
        if not scale > 1e-9 or n < 4:
            break
        cutoff = 4.685 * scale
        updated = []
        for i in range(n):
            u = residuals[i] / cutoff
            if abs(u) >= 1.0:
                updated.append(weights[i] * 0.001)
            else:
                keep = 1.0 - u * u
                updated.append(weights[i] * keep * keep)
        weights = updated
        if not solve():
            break

    total_w = sum(weights)
    if not total_w > 0:
        return None
    mean_y = sum(weights[i] * ys[i] for i in range(n)) / total_w

    sst = 0.0
    sse = 0.0
    for i in range(n):
        dev = ys[i] - mean_y
        sst += weights[i] * dev * dev
        sse += weights[i] * residuals[i] * residuals[i]
    quality = clamp(1.0 - sse / sst, 0.0, 1.0) if sst > 1e-12 else 0.0

    mean_t = sum(weights[i] * xs[i] for i in range(n)) / total_w
    centered_sxx = 0.0
    for i in range(n):
        dt = xs[i] - mean_t
        centered_sxx += weights[i] * dt * dt
    if centered_sxx > 1e-12:
        se = math.sqrt(sse / max(1, n - 2) / centered_sxx)
    else:
        se = 0.0

    return {"slope": slope, "quality": quality, "n": n, "se": se}


def robust_move_scale(moves):
    """MAD, or a trimmed RMS when rounded rates make most deviations zero.

    The trimmed RMS drops the single largest move so one bad page cannot set
    a trophy's volatility.
    """
    if not moves:
        return 0.0
    center = median(moves)
    mad = 1.4826 * median([abs(value - center) for value in moves])
    if mad > 1e-12:
        return mad

    squares = sorted(value * value for value in moves)
    if len(squares) >= 3:
        squares.pop()
    return math.sqrt(sum(squares) / len(squares))


def quantile_row(y_med, sigma):
    previous = 0.0
    row = []
    for z in QUANTILE_Z:
        y = y_med + z * sigma
        try:
            p = 100.0 / (1.0 + math.exp(-y))
        except OverflowError:
            p = 0.0 if y < 0 else 100.0
        if not math.isfinite(p):
            p = 0.0 if z < 0 else 100.0
        p = clamp(p, 0.0, 100.0)
        rounded = round4(p)
        if not math.isfinite(rounded):
            rounded = previous
        rounded = clamp(rounded, previous, 100.0)
        previous = rounded
        row.append(rounded)
    return row


def forecast_case(case):
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        trophies = []
    history = case.get("history")
    if not isinstance(history, list):
        history = []
    pages = collapse_pages(history)

    horizon_minutes = case.get("horizon_minutes")
    if not is_number(horizon_minutes):
        horizon_minutes = 0
    horizon_days = max(0.0, horizon_minutes / 1440.0)
    first_time = pages[0]["time"] if pages else 0.0

    # One observed series per trophy. Null pages are skipped, so the last
    # observation is the nearest earlier refresh when the final read lacked
    # the trophy. If a trophy never appears at all, the floor is used.
    series = []
    last_rates = []
    for trophy_index in range(len(trophies)):
        obs = []
        last_rate = None
        for page in pages:
            rate = page["rates"][trophy_index] if trophy_index < len(page["rates"]) else None
            if not is_number(rate):
                continue
            last_rate = rate
            obs.append(((page["time"] - first_time) / DAY_MS, to_logit(rate)))
        series.append(obs)
        last_rates.append(last_rate)

    raw_fits = [fit_robust_line(obs) for obs in series]

    tier_slopes = {}
    all_slopes = []
    for index, trophy in enumerate(trophies):
        fit = raw_fits[index]
        if fit is None:
            continue
        all_slopes.append(fit["slope"])
        tier = trophy.get("tier") if isinstance(trophy, dict) else None
        if not isinstance(tier, str):
            tier = ""
        tier_slopes.setdefault(tier, []).append(fit["slope"])
    if len(all_slopes) >= 2:
        global_slope = median(all_slopes) * (len(all_slopes) / (len(all_slopes) + 2.0))
    else:
        global_slope = 0.0

    # One-refresh-ahead moves in y per square-root day. A trophy's own scale
    # needs three moves; with fewer it borrows a pooled tier or global scale.
    trophy_moves = []
    for obs in series:
        moves = []
        for i in range(1, len(obs)):
            dt = obs[i][0] - obs[i - 1][0]
            if not dt > 1e-9:
                continue
            moves.append((obs[i][1] - obs[i - 1][1]) / math.sqrt(dt))
        trophy_moves.append(moves)
    own_vol = [robust_move_scale(moves) if len(moves) >= 3 else 0.0 for moves in trophy_moves]

    tier_moves = {}
    all_moves = []
    for index, trophy in enumerate(trophies):
        tier = trophy.get("tier") if isinstance(trophy, dict) else None
        if not isinstance(tier, str):
            tier = ""
        bucket = tier_moves.setdefault(tier, [])
        bucket.extend(trophy_moves[index])
        all_moves.extend(trophy_moves[index])
    global_vol = max(robust_move_scale(all_moves), VOL_FLOOR) if len(all_moves) >= 3 else VOL_FLOOR

    rows = []
    emitted = set()
    for index, trophy in enumerate(trophies):
        if not isinstance(trophy, dict):
            continue
        trophy_id = trophy.get("id")
        if not is_number(trophy_id):
            continue
        if trophy_id in emitted:
            continue
        emitted.add(trophy_id)

        # Cumulative rarity saturates: the contract wants literal boundary
        # rows rather than values a hair inside 0 or 100.
        last_raw = last_rates[index]
        if is_number(last_raw) and last_raw >= 100:
            rows.append({"id": int(trophy_id), "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        if is_number(last_raw) and last_raw <= 0:
            rows.append({"id": int(trophy_id), "q": [0.0, 0.0, 0.0, 0.0, 0.0]})
            continue

        obs = series[index]
        fit = raw_fits[index]
        last_y = to_logit(RATE_FLOOR)
        if obs:
            last_y = obs[-1][1]
        y_med = last_y

        if fit is not None:
            tier = trophy.get("tier")
            if not isinstance(tier, str):
                tier = ""
            tier_list = tier_slopes.get(tier, [])
            tier_slope = global_slope
            if len(tier_list) >= 2:
                raw_tier = median(tier_list)
                strength = len(tier_list) / (len(tier_list) + 2.0)
                tier_slope = raw_tier * strength

            n_factor = clamp((fit["n"] - 2) / 6.0, 0.0, 1.0)
            quality = clamp(fit["quality"], 0.0, 1.0)
            t_ratio = abs(fit["slope"]) / max(fit["se"], 1e-9)
            signal = t_ratio / (t_ratio + 1.0)
            keep = 0.2 + 0.8 * math.sqrt(n_factor * quality * signal)
            tier_weight = clamp(
                0.6 * (1.0 - n_factor) + 0.25 * (1.0 - quality),
                0.0,
                0.8,
            )

            used = (1.0 - tier_weight) * fit["slope"] * keep + tier_weight * tier_slope
            used = clamp(used, -DAILY_SLOPE_CAP, DAILY_SLOPE_CAP)
            raw_move = used * horizon_days
            move = clamp(MOVE_CAP * math.tanh(raw_move / MOVE_CAP), -MOVE_CAP, MOVE_CAP)
            y_med = last_y + DAMPING * move

        tier = trophy.get("tier")
        if not isinstance(tier, str):
            tier = ""
        pooled_tier_moves = tier_moves.get(tier, [])
        vol = own_vol[index] if own_vol[index] > 0 else 0.0
        if not vol > 0:
            if len(pooled_tier_moves) >= 3:
                vol = robust_move_scale(pooled_tier_moves)
            else:
                vol = global_vol
        vol = max(vol, VOL_FLOOR)

        sigma = math.sqrt(vol * vol * horizon_days + TIMING_SIGMA * TIMING_SIGMA)
        sigma = clamp(sigma, VOL_FLOOR, SIGMA_CAP)

        rows.append({"id": int(trophy_id), "q": quantile_row(y_med, sigma)})

    return rows


def encode_rows(rows):
    """Compact JSON with the TypeScript twin's integer formatting."""
    normalized = []
    for row in rows:
        q = []
        for value in row["q"]:
            if float(value).is_integer():
                q.append(int(value))
            else:
                q.append(value)
        normalized.append({"id": int(row["id"]), "q": q})
    return json.dumps({"trophies": normalized}, separators=(",", ":"))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(name for name in os.listdir(INPUT_DIR) if name.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        rows = forecast_case(case)
        payload = encode_rows(rows)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
