"""
sdz_oi.py — test open interest against the pre-registered design.

Reuses the permutation machinery already validated in sdz_delivery.py rather
than reimplementing it, so both studies are scored by the same code.

See PREREGISTRATION_oi.md. The bar is stated on ABSOLUTE net spread: the two
folk readings of a short build-up at support predict opposite signs, so no
direction was claimed in advance and whichever appears is the finding.

    python sdz_oi.py --touches touches.csv --oi data/nse_oi.csv.gz

Self-test:
    python sdz_oi.py --touches touches.csv --selftest noise
    python sdz_oi.py --touches touches.csv --selftest planted
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from sdz_delivery import BAR_MONO, BAR_NET, p, perm_spread, spread_of, tercile

FACTORS = [
    ("oi_chg_rel", "OI change / trailing 20d average OI"),
    ("oi_chg_pct", "OI change as % of yesterday's OI"),
    ("oi_level_z", "OI / trailing 20d mean OI (how crowded positioning is)"),
    ("pre_oi_chg_rel", "OI change / avg, on the bar BEFORE the touch"),
]


def build_factors(oi: pd.DataFrame) -> pd.DataFrame:
    """Trailing windows end at the PREVIOUS bar, so the touch bar is never in
    its own denominator."""
    d = oi.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)

    prev_oi = g["oi"].shift(1)
    avg20 = g["oi"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())

    # recompute the change rather than trusting the vendor column, which is
    # per-contract and does not survive summing across expiries at a rollover
    chg = d["oi"] - prev_oi

    d["oi_chg_rel"] = chg / avg20.replace(0, np.nan)
    d["oi_chg_pct"] = chg / prev_oi.replace(0, np.nan) * 100.0
    d["oi_level_z"] = d["oi"] / avg20.replace(0, np.nan)
    d["pre_oi_chg_rel"] = g["oi_chg_rel"].shift(1)

    # a rollover week can move summed OI violently; clip the extreme tails so a
    # handful of expiry artefacts cannot own a whole quintile
    for c in ("oi_chg_rel", "oi_chg_pct", "pre_oi_chg_rel"):
        lo, hi = d[c].quantile([0.001, 0.999])
        d[c] = d[c].clip(lo, hi)

    d["date"] = d["date"].dt.date.astype(str)
    return d[["date", "symbol", "oi_chg_rel", "oi_chg_pct",
              "oi_level_z", "pre_oi_chg_rel"]]


def buildup_table(d: pd.DataFrame, out: list) -> None:
    """DESCRIPTIVE ONLY — declared in advance as not a scored test."""
    if "px_dir" not in d.columns or "oi_chg_rel" not in d.columns:
        return
    m = d[d["oi_chg_rel"].notna() & d["px_dir"].notna()]
    if len(m) < 400:
        return
    p(out, "\n#### four-state build-up (descriptive, not scored)\n")
    p(out, "| price on touch bar | OI | reading | n | reversed |")
    p(out, "|---|---|---|---:|---:|")
    for pd_up in (False, True):
        for oi_up in (False, True):
            sel = m[((m["px_dir"] > 0) == pd_up) & ((m["oi_chg_rel"] > 0) == oi_up)]
            if len(sel) < 50:
                continue
            reading = {(False, True): "short build-up", (False, False): "long unwinding",
                       (True, True): "long build-up", (True, False): "short covering"}[(pd_up, oi_up)]
            p(out, f"| {'up' if pd_up else 'down'} | {'up' if oi_up else 'down'} | "
                   f"{reading} | {len(sel):,} | {sel['label'].mean()*100:.1f}% |")
    p(out, "\n> Read the OI columns only, never the price rows. A touch bar that "
           "closes up has already travelled part of the way to the upper barrier, "
           "so the up/down gap here is the same label-geometry artefact that killed "
           "the Hold Score — it appears at full strength on random data. The OI "
           "split WITHIN a price direction is the only comparison this table "
           "supports, and even that is descriptive: no weight may be taken from it.\n")


def test_side(df: pd.DataFrame, side: str, out: list) -> list[dict]:
    d = df[df["side"] == side]
    if len(d) < 500:
        return []
    y = d["label"].to_numpy(float)
    p(out, f"\n## side = {side}   n = {len(d):,}   base rate = {y.mean()*100:.2f}%\n")

    dep = tercile(d["f6_penetration"].to_numpy(float))
    vol = tercile(d["f7_touch_volume"].to_numpy(float))
    sym, dt = d["symbol"].to_numpy(), d["touch_date"].to_numpy()
    yr = pd.to_datetime(d["touch_date"]).dt.year.to_numpy()

    results = []
    for fid, desc in FACTORS:
        if fid not in d.columns:
            continue
        v = d[fid].to_numpy(float)
        if int(np.isfinite(v).sum()) < 500:
            p(out, f"### {fid} — too few rows with data, skipped\n")
            continue

        raw, mono, buckets = spread_of(v, y)
        c1m, c1s = perm_spread(v, y, sym)
        c2m, c2s = perm_spread(v, y, dt)
        # the bar is on |net|, so take whichever control is harder to beat in
        # the direction the real spread actually points
        worst = max(c1m, c2m) if raw >= 0 else min(c1m, c2m)
        net = raw - worst
        z1 = (raw - c1m) / c1s if c1s and c1s > 0 else np.nan
        z2 = (raw - c2m) / c2s if c2s and c2s > 0 else np.nan

        p(out, f"### {fid} — {desc}\n")
        p(out, "| bucket | n | reversal rate |")
        p(out, "|---:|---:|---:|")
        for k, n, r in buckets:
            p(out, f"| {k} | {n:,} | {r:.1f}% |")
        p(out, "")
        p(out, f"- raw spread        **{raw:+.2f}** points, monotonicity {mono:.2f}")
        p(out, f"- C1 within-stock   {c1m:+.2f} ± {c1s:.2f}   (z = {z1:+.1f})")
        p(out, f"- C2 within-date    {c2m:+.2f} ± {c2s:.2f}   (z = {z2:+.1f})")
        p(out, f"- **net of the worse control: {net:+.2f}**  (bar is |net| >= {BAR_NET:.0f})")

        cond = {}
        for name, band in (("depth", dep), ("volume", vol)):
            got = []
            for t in range(3):
                m = band == t
                if m.sum() < 150:
                    got.append(np.nan)
                    continue
                s_, _, _ = spread_of(v[m], y[m])
                got.append(s_)
            cond[name] = got
            lab = ["low", "mid", "high"]
            p(out, f"- C3 within {name} terciles: " + "  ".join(
                f"{lab[i]} {g:+.1f}" if g == g else f"{lab[i]} n/a" for i, g in enumerate(got)))

        ys = []
        for u in sorted(set(yr)):
            m = yr == u
            if m.sum() < 300:
                continue
            s_, _, _ = spread_of(v[m], y[m])
            ys.append((u, int(m.sum()), s_))
        if ys:
            p(out, "- C4 by year: " + "  ".join(f"{u} {s_:+.1f} (n={n})" for u, n, s_ in ys))
        agree = bool(ys) and (all(s_ > 0 for _, _, s_ in ys) or all(s_ < 0 for _, _, s_ in ys))

        c3ok = sum(1 for g in cond.get("depth", [])
                   if g == g and abs(g) >= BAR_NET / 2) >= 2
        passes = (abs(net) >= BAR_NET) and (mono == mono and mono >= BAR_MONO) and c3ok and agree
        why = []
        if abs(net) < BAR_NET:
            why.append(f"|net| {abs(net):.1f} < {BAR_NET:.0f}")
        if not (mono == mono and mono >= BAR_MONO):
            why.append(f"monotonicity {mono:.2f} < {BAR_MONO}")
        if not c3ok:
            why.append("dies inside depth bands")
        if not agree:
            why.append("sign flips across years")
        verdict = "SURVIVES" if passes else "does not survive"
        p(out, f"- **verdict: {verdict}**" + (f" — {', '.join(why)}" if why else ""))
        p(out, "")

        results.append(dict(side=side, factor=fid, n=int(np.isfinite(v).sum()),
                            raw=raw, mono=mono, c1=c1m, c2=c2m, net=net, verdict=verdict))

    buildup_table(d, out)
    return results


def synth(t: pd.DataFrame, mode: str, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(t)
    base = rng.normal(0, 1, n)
    if mode == "planted":
        base = base + np.where(t["side"].to_numpy() == "D",
                               (t["label"].to_numpy() - 0.5) * 1.6, 0.0)
    o = t[["touch_date", "nse_symbol"]].copy()
    o.columns = ["date", "symbol"]
    o["oi_chg_rel"] = base
    o["oi_chg_pct"] = base * 3
    o["oi_level_z"] = 1 + base * 0.05
    o["pre_oi_chg_rel"] = rng.permutation(base)
    return o.drop_duplicates(subset=["date", "symbol"], keep="first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches", default="touches.csv")
    ap.add_argument("--oi", default="data/nse_oi.csv.gz")
    ap.add_argument("--prices", default="data/nse_daily.csv.gz",
                    help="only used for the descriptive build-up table")
    ap.add_argument("--selftest", choices=["noise", "planted"])
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    t = pd.read_csv(a.touches)
    if "nse_symbol" not in t.columns:
        t["nse_symbol"] = t["symbol"].str.replace(r"\.NS$", "", regex=True)
    # price direction ON THE TOUCH BAR ITSELF (close vs previous close), for the
    # descriptive build-up table. Not the side — a demand touch can still close up.
    t["px_dir"] = np.nan
    try:
        px = pd.read_csv(a.prices)
        px.columns = [c.strip().lower().replace(" ", "_") for c in px.columns]
        px["date"] = pd.to_datetime(px["date"]).dt.date.astype(str)
        px = px.sort_values(["symbol", "date"])
        px["ret"] = px.groupby("symbol", sort=False)["close"].diff()
        px["nse_symbol"] = px["symbol"].str.replace(r"\.NS$", "", regex=True)
        t = t.merge(px[["date", "nse_symbol", "ret"]],
                    left_on=["touch_date", "nse_symbol"],
                    right_on=["date", "nse_symbol"], how="left")
        t["px_dir"] = np.sign(t["ret"])
        t = t.drop(columns=["date", "ret"])
    except FileNotFoundError:
        print(f"note: {a.prices} not found — build-up table will be skipped",
              file=sys.stderr)

    if a.selftest:
        fac = synth(t, a.selftest)
        header = f"# SELF-TEST ({a.selftest}) — fabricated data, not a result"
    else:
        try:
            raw = pd.read_csv(a.oi)
        except FileNotFoundError:
            print(f"missing {a.oi} — run fetch_oi.py on the GitHub Action first.",
                  file=sys.stderr)
            return 2
        fac = build_factors(raw)
        header = ("# Open interest at zone touches\n\n"
                  f"{len(t):,} touches. Design fixed in advance — "
                  "see PREREGISTRATION_oi.md.")

    m = t.merge(fac, left_on=["touch_date", "nse_symbol"],
                right_on=["date", "symbol"], how="left", suffixes=("", "_f"))
    if not a.selftest:
        header += f"\n\n{m['oi_chg_rel'].notna().mean()*100:.1f}% matched to an OI record."

    out: list[str] = []
    p(out, header)
    rows = []
    for side in ("D", "S"):
        rows += test_side(m, side, out)

    if rows:
        r = pd.DataFrame(rows)
        p(out, "\n## Summary\n")
        p(out, "| side | factor | n | raw | C1 | C2 | net | verdict |")
        p(out, "|---|---|---:|---:|---:|---:|---:|---|")
        for _, x in r.iterrows():
            p(out, f"| {x['side']} | {x['factor']} | {x['n']:,} | {x['raw']:+.2f} | "
                   f"{x['c1']:+.2f} | {x['c2']:+.2f} | **{x['net']:+.2f}** | {x['verdict']} |")

    if a.report:
        with open(a.report, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"\nwrote {a.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
