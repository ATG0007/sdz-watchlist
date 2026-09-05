"""
sdz_core.py — zone detection + feature extraction, ported from SDZ Pro v3.2 Pine.

The rule here is absolute: every feature is computed using bars up to and
including the touch bar, never one bar further. If you add a feature, check it
against that rule before you run anything. A single leaked future bar produces a
beautiful result that will not survive contact with a live market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters — mirror the Pine defaults. Change them here, not in the caller.
# ---------------------------------------------------------------------------
P = dict(
    atr_len=14,
    base_body_ratio=0.6,
    base_range_atr=1.3,
    min_base=1,
    max_base=6,
    leg_atr=0.8,
    vol_len=20,
    min_score=50,
    base_score_cap=75.0,
    sweep_lookback=10,
    sweep_bonus=15.0,
    pool_lookback=30,
    pool_tol_atr=0.15,
    pool_bonus=10.0,
    merge_proximity_atr=0.3,
    max_zones_each=3,
    max_zone_age=150,
    retest_buffer_atr=0.5,
    # scorecard feature settings
    eff_lookback=10,
    struct_lookback=20,
    # label (barrier race) — FIX THESE BEFORE LOOKING AT RESULTS
    target_atr=1.00,     # upper barrier = zone_top + this * ATR
    stop_atr=0.25,       # lower barrier = zone_bottom - this * ATR
    horizon=15,          # bars
    warmup=150,          # bars skipped at the start of each series (EMA/ATR settling)
)


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder smoothing — what Pine's ta.rma / ta.atr actually use."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(tr, length)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add every derived series the detector and the features need."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    df["atr"] = atr(df, P["atr_len"])
    df["avg_vol"] = df["volume"].rolling(P["vol_len"]).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["lo_sweep"] = df["low"].rolling(P["sweep_lookback"]).min()
    df["hi_sweep"] = df["high"].rolling(P["sweep_lookback"]).max()
    df["lo_struct"] = df["low"].rolling(P["struct_lookback"]).min()
    df["hi_struct"] = df["high"].rolling(P["struct_lookback"]).max()

    # approach efficiency: signed net move / total distance travelled
    step = (df["close"] - df["close"].shift(1)).abs()
    abs_path = step.rolling(P["eff_lookback"]).sum()
    net_path = df["close"] - df["close"].shift(P["eff_lookback"])
    df["eff_signed"] = np.where(abs_path > 0, net_path / abs_path, 0.0)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / rng).fillna(0.0)
    df["is_base"] = (df["body_ratio"] <= P["base_body_ratio"]) & (
        (df["high"] - df["low"]) <= P["base_range_atr"] * df["atr"]
    )
    df["bull_leg"] = (df["close"] - df["open"]) >= P["leg_atr"] * df["atr"]
    df["bear_leg"] = (df["open"] - df["close"]) >= P["leg_atr"] * df["atr"]
    return df


# ---------------------------------------------------------------------------
# hold score — identical maths to the Pine
# ---------------------------------------------------------------------------
def hold_score(row, is_demand: bool) -> float:
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.0
    if is_demand:
        wick = min(row["open"], row["close"]) - row["low"]
        close_strength = (row["close"] - row["low"]) / rng
        colour = 15.0 if row["close"] > row["open"] else 0.0
    else:
        wick = row["high"] - max(row["open"], row["close"])
        close_strength = (row["high"] - row["close"]) / rng
        colour = 15.0 if row["close"] < row["open"] else 0.0
    vr = row["volume"] / row["avg_vol"] if row["avg_vol"] and row["avg_vol"] > 0 else 1.0
    vol_pts = max(0.0, min(25.0, (vr - 1.0) * 25.0))
    return min(100.0, wick / rng * 35.0 + close_strength * 25.0 + vol_pts + colour)


def _has_equal_extremes(arr: np.ndarray, tol: float, want_high: bool) -> bool:
    if len(arr) == 0 or not np.isfinite(tol):
        return False
    ref = arr.max() if want_high else arr.min()
    return int(np.sum(np.abs(arr - ref) <= tol)) >= 2


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
def extract_touches(df: pd.DataFrame, symbol: str, index_ret: pd.Series | None = None) -> list[dict]:
    """Walk the series bar by bar, maintain live zones, emit a row per touch."""
    d = prepare(df)
    n_bars = len(d)
    if n_bars < P["warmup"] + 40:
        return []

    H, L, C, O = d["high"].values, d["low"].values, d["close"].values, d["open"].values
    V, A = d["volume"].values, d["atr"].values
    AVGV = d["avg_vol"].values
    EMA20, EMA200 = d["ema20"].values, d["ema200"].values
    LO_SW, HI_SW = d["lo_sweep"].values, d["hi_sweep"].values
    LO_ST, HI_ST = d["lo_struct"].values, d["hi_struct"].values
    EFF = d["eff_signed"].values
    IS_BASE, BULL, BEAR = d["is_base"].values, d["bull_leg"].values, d["bear_leg"].values
    DATES = d["date"].values

    idx_ret = None
    if index_ret is not None:
        idx_ret = index_ret.reindex(pd.to_datetime(d["date"])).values

    demand: list[dict] = []   # each: top, bot, score, left, tested, touches
    supply: list[dict] = []
    rows: list[dict] = []

    start = max(P["warmup"], P["atr_len"] + P["pool_lookback"] + P["max_base"] + 5)

    for i in range(start, n_bars):
        atr_i = A[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # ---------------- detection on this (closed) bar ----------------
        if BULL[i] or BEAR[i]:
            is_demand = bool(BULL[i])
            n = 0
            for k in range(1, P["max_base"] + 1):
                if IS_BASE[i - k]:
                    n = k
                else:
                    break

            if n >= P["min_base"]:
                base = slice(i - n, i)
                zone_high = H[base].max()
                zone_low = L[base].min()
                body_top = np.maximum(O[base], C[base]).max()
                body_bot = np.minimum(O[base], C[base]).min()

                leg_body = abs(C[i] - O[i])
                leg_strength = leg_body / atr_i
                vol_ratio = V[i] / AVGV[i - 1] if AVGV[i - 1] and AVGV[i - 1] > 0 else 1.0

                ref = i - (n + 1)
                swept = (zone_low < LO_SW[ref]) if is_demand else (zone_high > HI_SW[ref])

                top = body_top if is_demand else zone_high
                bot = zone_low if is_demand else body_bot

                p0 = i - (n + 2)
                p1 = p0 - P["pool_lookback"] + 1
                pool = False
                if p1 >= 0:
                    seg = H[p1 : p0 + 1] if is_demand else L[p1 : p0 + 1]
                    pool = _has_equal_extremes(seg, P["pool_tol_atr"] * atr_i, is_demand)

                leg_score = min(40.0, leg_strength * 20.0)
                vol_score = max(0.0, min(30.0, (vol_ratio - 1.0) * 30.0))
                fresh = (P["max_base"] - n + 1.0) / P["max_base"] * 30.0
                base_score = min(P["base_score_cap"], leg_score + vol_score + fresh)
                score = min(100.0, round(base_score
                                         + (P["sweep_bonus"] if swept else 0.0)
                                         + (P["pool_bonus"] if pool else 0.0)))

                if score >= P["min_score"] and top > bot:
                    store = demand if is_demand else supply
                    merged = False
                    for z in store:
                        gap = 0.0
                        if top < z["bot"]:
                            gap = z["bot"] - top
                        elif z["top"] < bot:
                            gap = bot - z["top"]
                        if gap <= P["merge_proximity_atr"] * atr_i:
                            z["top"] = max(z["top"], top)
                            z["bot"] = min(z["bot"], bot)
                            z["score"] = max(z["score"], score)
                            z["left"] = min(z["left"], i - n)
                            z["tested"] = False
                            z["touches"] = 0
                            merged = True
                            break
                    if not merged:
                        store.append(dict(top=top, bot=bot, score=score,
                                          left=i - n, tested=False, touches=0,
                                          swept=bool(swept), pool=bool(pool), n_base=n))
                        if len(store) > P["max_zones_each"]:
                            store.remove(min(store, key=lambda z: z["score"]))

        # ---------------- age out ----------------
        demand = [z for z in demand if i - z["left"] <= P["max_zone_age"]]
        supply = [z for z in supply if i - z["left"] <= P["max_zone_age"]]

        # ---------------- mitigation + touches ----------------
        for is_demand, store in ((True, demand), (False, supply)):
            dead = []
            for z in store:
                if is_demand and C[i] < z["bot"]:
                    dead.append(z)
                    continue
                if (not is_demand) and C[i] > z["top"]:
                    dead.append(z)
                    continue

                # re-arm once price has left the zone by the buffer
                if z["tested"]:
                    left_zone = (L[i] > z["top"] + P["retest_buffer_atr"] * atr_i) if is_demand \
                        else (H[i] < z["bot"] - P["retest_buffer_atr"] * atr_i)
                    if left_zone:
                        z["tested"] = False

                touching = (L[i] <= z["top"] and H[i] >= z["bot"])
                if z["tested"] or not touching:
                    continue

                z["tested"] = True
                z["touches"] += 1

                row = _build_row(
                    d, i, z, is_demand, symbol, DATES, H, L, C, O, V, A, AVGV,
                    EMA20, EMA200, LO_ST, HI_ST, EFF, idx_ret, n_bars,
                )
                if row is not None:
                    rows.append(row)

            for z in dead:
                store.remove(z)

    return rows


def _build_row(d, i, z, is_demand, symbol, DATES, H, L, C, O, V, A, AVGV,
               EMA20, EMA200, LO_ST, HI_ST, EFF, idx_ret, n_bars):
    atr_i = A[i]
    zone_rng = max(z["top"] - z["bot"], 1e-9)
    hs = hold_score(d.iloc[i], is_demand)

    # ---- the nine scorecard features, all as of bar i ----
    f_eff = clamp(EFF[i] if is_demand else -EFF[i])
    new_extreme = (L[i] <= LO_ST[i - 1]) if is_demand else (H[i] >= HI_ST[i - 1])
    f_struct = -1.0 if new_extreme else 1.0
    ext_raw = ((EMA20[i] - C[i]) if is_demand else (C[i] - EMA20[i])) / atr_i
    f_ext = clamp(ext_raw / 2.0)
    f_touch = 1.0 if z["touches"] <= 1 else (0.0 if z["touches"] == 2 else -1.0)
    tape_up = C[i] > EMA20[i]
    counter = (not tape_up) if is_demand else tape_up
    f_counter = 1.0 if counter else -1.0
    depth = (z["top"] - L[i]) / zone_rng if is_demand else (H[i] - z["bot"]) / zone_rng
    f_depth = 1.0 - 2.0 * clamp(depth, 0.0, 1.0)
    vr = V[i] / AVGV[i] if AVGV[i] and AVGV[i] > 0 else 1.0
    f_vol = clamp(vr - 1.0)
    f_hold = clamp((hs - 50.0) / 50.0)
    ir = 0.0
    if idx_ret is not None and np.isfinite(idx_ret[i]):
        ir = clamp(idx_ret[i] / 0.01)
    f_idx = (-ir if is_demand else ir)

    # ---- label: barrier race, resolved on forward bars only ----
    if is_demand:
        up_barrier = z["top"] + P["target_atr"] * atr_i
        dn_barrier = z["bot"] - P["stop_atr"] * atr_i
    else:
        up_barrier = z["bot"] - P["target_atr"] * atr_i   # "favourable" = down
        dn_barrier = z["top"] + P["stop_atr"] * atr_i

    label, bars_to_res = 0, np.nan
    mfe, mae = -np.inf, np.inf
    end = min(i + P["horizon"], n_bars - 1)
    if i + 1 > n_bars - 1:
        return None
    for j in range(i + 1, end + 1):
        if is_demand:
            mfe = max(mfe, (H[j] - z["top"]) / atr_i)
            mae = min(mae, (L[j] - z["bot"]) / atr_i)
            if C[j] >= up_barrier:
                label, bars_to_res = 1, j - i
                break
            if C[j] <= dn_barrier:
                label, bars_to_res = 0, j - i
                break
        else:
            mfe = max(mfe, (z["bot"] - L[j]) / atr_i)
            mae = min(mae, (z["top"] - H[j]) / atr_i)
            if C[j] <= up_barrier:
                label, bars_to_res = 1, j - i
                break
            if C[j] >= dn_barrier:
                label, bars_to_res = 0, j - i
                break

    # not enough forward bars to resolve honestly -> drop the row
    if np.isnan(bars_to_res) and end < i + P["horizon"]:
        return None

    # ---- entry realism + things knowable BEFORE the touch bar -------------
    # depth measured in ATRs, not as a fraction of the zone: a wide zone is
    # mechanically easy to fill "only a little", which confounds the % version.
    depth_atr = ((z["top"] - L[i]) if is_demand else (H[i] - z["bot"])) / atr_i
    # where the touch bar closed relative to the near edge (+ = closed clear of it)
    close_pos_atr = ((C[i] - z["top"]) if is_demand else (z["bot"] - C[i])) / atr_i

    # pre-touch: every one of these is known at the PREVIOUS close or this open
    pre_drop = ((C[i - 4] - C[i - 1]) if is_demand else (C[i - 1] - C[i - 4])) / atr_i
    pre_dist = ((C[i - 1] - z["top"]) if is_demand else (z["bot"] - C[i - 1])) / atr_i
    pre_vol = V[i - 1] / AVGV[i - 1] if AVGV[i - 1] and AVGV[i - 1] > 0 else 1.0
    pre_gap = ((O[i] - C[i - 1]) if is_demand else (C[i - 1] - O[i])) / atr_i
    pre_run = sum(1 for k in range(1, 6)
                  if (C[i - k] < C[i - k - 1] if is_demand else C[i - k] > C[i - k - 1]))

    # ---- the trade you could actually take: see the close, enter next open ----
    entry = O[i + 1]
    if is_demand:
        tgt_px, stop_px = z["top"] + P["target_atr"] * atr_i, z["bot"] - P["stop_atr"] * atr_i
        risk_px = entry - stop_px
    else:
        tgt_px, stop_px = z["bot"] - P["target_atr"] * atr_i, z["top"] + P["stop_atr"] * atr_i
        risk_px = stop_px - entry
    r_real, won_real = np.nan, np.nan
    if risk_px > 0:
        reward_px = (tgt_px - entry) if is_demand else (entry - tgt_px)
        won_real, r_real = 0, None
        for j in range(i + 1, end + 1):
            if is_demand:
                if C[j] >= tgt_px:
                    won_real, r_real = 1, reward_px / risk_px
                    break
                if C[j] <= stop_px:
                    won_real, r_real = 0, -1.0
                    break
            else:
                if C[j] <= tgt_px:
                    won_real, r_real = 1, reward_px / risk_px
                    break
                if C[j] >= stop_px:
                    won_real, r_real = 0, -1.0
                    break
        if r_real is None:   # timed out - mark to market, don't pretend it was flat
            r_real = ((C[end] - entry) if is_demand else (entry - C[end])) / risk_px

    return dict(
        symbol=symbol,
        touch_date=pd.Timestamp(DATES[i]).date().isoformat(),
        side="D" if is_demand else "S",
        zone_top=round(float(z["top"]), 4),
        zone_bot=round(float(z["bot"]), 4),
        zone_score=float(z["score"]),
        touch_number=int(z["touches"]),
        days_since_formation=int(i - z["left"]),
        # --- the nine ---
        f1_approach_impulse=round(float(f_eff), 4),
        f2_new_extreme=float(f_struct),
        f3_extension=round(float(f_ext), 4),
        f4_touch_number=float(f_touch),
        f5_counter_trend=float(f_counter),
        f6_penetration=round(float(f_depth), 4),
        f7_touch_volume=round(float(f_vol), 4),
        f8_hold_score=round(float(f_hold), 4),
        f9_index_context=round(float(f_idx), 4),
        # --- extras worth testing ---
        hold_score_raw=round(float(hs), 2),
        zone_width_atr=round(float(zone_rng / atr_i), 4),
        dist_200ema_atr=round(float((C[i] - EMA200[i]) / atr_i), 4),
        atr_pct=round(float(atr_i / C[i] * 100.0), 4),
        zone_swept=int(z.get("swept", False)),
        zone_pool=int(z.get("pool", False)),
        # --- outcome ---
        label=int(label),
        bars_to_resolution=float(bars_to_res) if not np.isnan(bars_to_res) else float(P["horizon"]),
        depth_atr=round(float(depth_atr), 4),
        close_pos_atr=round(float(close_pos_atr), 4),
        pre_drop_atr=round(float(pre_drop), 4),
        pre_dist_atr=round(float(pre_dist), 4),
        pre_vol_ratio=round(float(pre_vol), 4),
        pre_gap_atr=round(float(pre_gap), 4),
        pre_down_days=int(pre_run),
        entry_next_open=round(float(entry), 4),
        entry_slip_atr=round(float(((entry - z["top"]) if is_demand else (z["bot"] - entry)) / atr_i), 4),
        risk_real_atr=round(float(risk_px / atr_i), 4) if risk_px > 0 else 0.0,
        won_real=int(won_real) if won_real == won_real else -1,
        r_real=round(float(r_real), 4) if r_real == r_real else 0.0,
        mfe_atr=round(float(mfe), 4) if np.isfinite(mfe) else 0.0,
        mae_atr=round(float(mae), 4) if np.isfinite(mae) else 0.0,
    )
