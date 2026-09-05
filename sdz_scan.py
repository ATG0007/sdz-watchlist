"""
sdz_scan.py — which stocks are sitting near a live demand zone right now.

Runs the same detector the whole study used, right up to the last bar in the
data, and reports the demand zones that are still alive (never closed through)
together with the trade geometry the study says actually matters: how far the
stop has to sit, and therefore how big the position can be.

This is a watchlist, not a set of signals. The study found no filter that beat a
null model, so nothing here is ranked by "probability". It is ranked by how close
price is and how tight the stop would be.

Usage
-----
    python sdz_scan.py --prices nse_20y.csv.gz --max-atr 1.5 --out watchlist.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sdz_core import P, prepare, _has_equal_extremes

ZONE_LIFE = 300          # bars a zone stays on the watchlist if never touched
MAX_PER_SIDE = 8         # keep more than the trading default so nothing good is evicted


def live_demand_zones(d: pd.DataFrame):
    """Walk the series once and return the demand zones still alive at the end."""
    H, L, C, O, V, A = (d.high.values, d.low.values, d.close.values,
                        d.open.values, d.volume.values, d.atr.values)
    AVGV, LO_SW = d.avg_vol.values, d.lo_sweep.values
    IS_BASE, BULL = d.is_base.values, d.bull_leg.values
    n = len(d)
    zones: list[dict] = []

    for i in range(max(60, P["atr_len"] + P["pool_lookback"] + P["max_base"] + 5), n):
        atr = A[i]
        if not np.isfinite(atr) or atr <= 0:
            continue

        if BULL[i]:
            nb = 0
            for k in range(1, P["max_base"] + 1):
                if IS_BASE[i - k]:
                    nb = k
                else:
                    break
            if nb >= P["min_base"]:
                base = slice(i - nb, i)
                body_top = np.maximum(O[base], C[base]).max()
                zone_low = L[base].min()
                leg_strength = abs(C[i] - O[i]) / atr
                vr = V[i] / AVGV[i - 1] if AVGV[i - 1] and AVGV[i - 1] > 0 else 1.0
                ref = i - (nb + 1)
                swept = zone_low < LO_SW[ref]
                p0 = i - (nb + 2)
                p1 = p0 - P["pool_lookback"] + 1
                pool = (_has_equal_extremes(H[p1:p0 + 1], P["pool_tol_atr"] * atr, True)
                        if p1 >= 0 else False)
                score = min(100.0, round(
                    min(P["base_score_cap"],
                        min(40.0, leg_strength * 20.0)
                        + max(0.0, min(30.0, (vr - 1.0) * 30.0))
                        + (P["max_base"] - nb + 1.0) / P["max_base"] * 30.0)
                    + (P["sweep_bonus"] if swept else 0.0)
                    + (P["pool_bonus"] if pool else 0.0)))
                if score >= P["min_score"] and body_top > zone_low:
                    merged = False
                    for z in zones:
                        gap = (z["bot"] - body_top if body_top < z["bot"] else
                               zone_low - z["top"] if z["top"] < zone_low else 0.0)
                        if gap <= P["merge_proximity_atr"] * atr:
                            z.update(top=max(z["top"], body_top),
                                     bot=min(z["bot"], zone_low),
                                     score=max(z["score"], score),
                                     left=min(z["left"], i - nb), touches=0)
                            merged = True
                            break
                    if not merged:
                        zones.append(dict(top=body_top, bot=zone_low, score=score,
                                          left=i - nb, touches=0, tested=False))
                        if len(zones) > MAX_PER_SIDE:
                            zones.remove(min(zones, key=lambda z: z["score"]))

        # kill anything price has closed through, and retire the very old
        zones = [z for z in zones if C[i] >= z["bot"] and i - z["left"] <= ZONE_LIFE]
        for z in zones:
            if z["tested"] and L[i] > z["top"] + P["retest_buffer_atr"] * atr:
                z["tested"] = False
            if not z["tested"] and L[i] <= z["top"] and H[i] >= z["bot"]:
                z["tested"] = True
                z["touches"] += 1
                z["last_touch"] = i
    return zones, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--max-atr", type=float, default=1.5,
                    help="how far above the zone top price may sit, in ATRs")
    ap.add_argument("--capital", type=float, default=10000.0,
                    help="rupees risked per trade, for the position-size column")
    ap.add_argument("--adjust", action="store_true",
                    help="dividend-adjust prices. OFF by default so the levels match a "
                         "TradingView chart, which does not dividend-adjust NSE stocks.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = pd.read_csv(args.prices)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw = raw.rename(columns={"adj close": "adj_close"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).drop_duplicates(subset=["date", "symbol"])
    last_date = raw["date"].max()

    rows = []
    stale = []
    for sym, g in raw.groupby("symbol"):
        if str(sym).startswith("^"):
            continue
        name = str(sym).replace(".NS", "")
        g = g.sort_values("date")
        if g["date"].max() < last_date - pd.Timedelta(days=5):
            stale.append(name)
            continue
        # only if asked: TradingView shows split-adjusted, dividend-UNadjusted prices,
        # and on high-dividend names the two differ by several percent
        if args.adjust and "adj_close" in g.columns:
            k = (g["adj_close"] / g["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            g = g.assign(**{c: g[c] * k for c in ("open", "high", "low", "close")})
        d = prepare(g)
        if len(d) < 300:
            continue
        zones, n = live_demand_zones(d)
        if not zones:
            continue
        i = n - 1
        close, atr = d.close.values[i], d.atr.values[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        for z in zones:
            gap_atr = (close - z["top"]) / atr           # <0 means price is inside it
            if gap_atr > args.max_atr or close < z["bot"]:
                continue
            stop = z["bot"] - 0.25 * atr
            risk = z["top"] - stop
            rows.append(dict(
                symbol=name, close=round(close, 2),
                zone_top=round(z["top"], 2), zone_bot=round(z["bot"], 2),
                gap_atr=round(gap_atr, 2),
                gap_pct=round((close - z["top"]) / close * 100, 2),
                inside=int(gap_atr <= 0),
                width_atr=round((z["top"] - z["bot"]) / atr, 2),
                stop=round(stop, 2),
                risk_pct=round(risk / z["top"] * 100, 2),
                shares=int(args.capital / risk) if risk > 0 else 0,
                target_2R=round(z["top"] + 2 * risk, 2),
                age_bars=int(i - z["left"]), touches=int(z["touches"]),
                atr_pct=round(atr / close * 100, 2), score=int(z["score"])))

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing within range")
        return
    # closest first, then tightest stop
    df = df.sort_values(["gap_atr", "risk_pct"]).reset_index(drop=True)
    if args.out:
        df.to_csv(args.out, index=False)

    print(f"\ndata to {last_date.date()}   |   {df.symbol.nunique()} stocks with a live "
          f"demand zone within {args.max_atr} ATR")
    if stale:
        print(f"skipped {len(stale)} symbols with stale data: {', '.join(stale[:8])}")
    print("=" * 118)
    print(f"{'stock':<13}{'close':>9}{'zone':>19}{'gap':>13}{'stop':>9}"
          f"{'risk':>8}{'2R target':>11}{'width':>7}{'age':>6}{'tch':>5}")
    print("-" * 118)
    for r in df.itertuples():
        gap = "INSIDE" if r.inside else f"{r.gap_atr:.2f}A/{r.gap_pct:.1f}%"
        print(f"{r.symbol:<13}{r.close:>9.2f}{r.zone_bot:>10.2f}-{r.zone_top:<8.2f}"
              f"{gap:>13}{r.stop:>9.2f}{r.risk_pct:>7.1f}%{r.target_2R:>11.2f}"
              f"{r.width_atr:>7.2f}{r.age_bars:>6}{r.touches:>5}")
    print("-" * 118)
    print(f"gap = how far above the zone top price closed (A = ATRs). risk = zone top to "
          f"stop, as % of entry.")
    print(f"age = bars since the zone formed. tch = times it has already been tested.")


if __name__ == "__main__":
    main()
