"""
sdz_delivery.py — test delivery percentage against the pre-registered design.

The random-walk null model used everywhere else in this study does not apply:
delivery percentage has no synthetic analogue, and inventing one would measure
the invention. It is replaced by two permutation controls (C1 within-stock,
C2 within-date), a conditional test against the two known survivors (C3), and
year-by-year sign consistency (C4). See PREREGISTRATION_delivery.md.

    python sdz_delivery.py --touches touches.csv --delivery data/nse_delivery.csv.gz

Self-test — proves the harness is honest before real data touches it:

    python sdz_delivery.py --touches touches.csv --selftest noise    # must find nothing
    python sdz_delivery.py --touches touches.csv --selftest planted  # must find the plant
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

NBUCKET = 5
NPERM = 300
BAR_NET = 10.0          # pre-registered: net spread must reach this
BAR_MONO = 0.60

FACTORS = [
    ("deliv_pct", "delivery % on the touch bar"),
    ("deliv_pct_rel", "delivery % / own trailing 20d median"),
    ("deliv_qty_rel", "delivered qty / own trailing 20d average"),
    ("pre_deliv_pct_rel", "delivery % / median, on the bar BEFORE the touch"),
]


# ---------------------------------------------------------------------------
# factor construction
# ---------------------------------------------------------------------------
def build_factors(dlv: pd.DataFrame) -> pd.DataFrame:
    """Per symbol-day factors. Trailing windows END AT THE PREVIOUS BAR, so the
    touch bar never appears in its own denominator."""
    d = dlv.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)

    med20 = g["deliv_pct"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    avg20 = g["deliv_qty"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())

    d["deliv_pct_rel"] = d["deliv_pct"] / med20.replace(0, np.nan)
    d["deliv_qty_rel"] = d["deliv_qty"] / avg20.replace(0, np.nan)
    d["pre_deliv_pct_rel"] = g["deliv_pct_rel"].shift(1)
    d["date"] = d["date"].dt.date.astype(str)
    return d[["date", "symbol", "deliv_pct", "deliv_pct_rel",
              "deliv_qty_rel", "pre_deliv_pct_rel"]]


# ---------------------------------------------------------------------------
# spread machinery
# ---------------------------------------------------------------------------
def _buckets(v: np.ndarray, n: int = NBUCKET) -> np.ndarray:
    """Quintile index, or -1 where the factor is missing."""
    out = np.full(len(v), -1, dtype=int)
    ok = np.isfinite(v)
    if ok.sum() < n * 20:
        return out
    try:
        q = pd.qcut(pd.Series(v[ok]), n, labels=False, duplicates="drop")
    except ValueError:
        return out
    out[ok] = q.to_numpy()
    return out


def spread_of(v: np.ndarray, y: np.ndarray) -> tuple[float, float, list]:
    """Top bucket minus bottom bucket hit rate, in points, plus monotonicity."""
    b = _buckets(v)
    ok = b >= 0
    if not ok.any():
        return np.nan, np.nan, []
    hi = int(b[ok].max())
    rates, ns = [], []
    for k in range(hi + 1):
        m = b == k
        ns.append(int(m.sum()))
        rates.append(float(y[m].mean() * 100) if m.sum() else np.nan)
    sp = rates[-1] - rates[0]
    steps = np.diff(rates)
    mono = float(np.mean(np.sign(steps) == np.sign(sp))) if len(steps) and sp == sp else np.nan
    return sp, mono, list(zip(range(hi + 1), ns, rates))


def perm_spread(v: np.ndarray, y: np.ndarray, key: np.ndarray,
                nperm: int = NPERM, seed: int = 7) -> tuple[float, float]:
    """Mean and sd of the spread when the factor is permuted WITHIN each key
    group — stock for C1, date for C2."""
    rng = np.random.default_rng(seed)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    bounds = list(zip(starts, np.r_[starts[1:], len(ks)]))
    out = []
    vv = v.copy()
    for _ in range(nperm):
        p = vv[order]
        for a, b in bounds:
            if b - a > 1:
                p[a:b] = rng.permutation(p[a:b])
        shuffled = np.empty_like(vv)
        shuffled[order] = p
        sp, _, _ = spread_of(shuffled, y)
        if sp == sp:
            out.append(sp)
    if not out:
        return np.nan, np.nan
    return float(np.mean(out)), float(np.std(out))


def tercile(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), -1, dtype=int)
    ok = np.isfinite(v)
    if ok.sum() < 60:
        return out
    try:
        out[ok] = pd.qcut(pd.Series(v[ok]), 3, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        pass
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def test_side(df: pd.DataFrame, side: str, out) -> list[dict]:
    d = df[df["side"] == side]
    if len(d) < 500:
        return []
    y = d["label"].to_numpy(float)
    p(out, f"\n## side = {side}   n = {len(d):,}   base rate = {y.mean()*100:.2f}%\n")

    dep = tercile(d["f6_penetration"].to_numpy(float))    # +1 = shallow
    vol = tercile(d["f7_touch_volume"].to_numpy(float))
    sym = d["symbol"].to_numpy()
    dt = d["touch_date"].to_numpy()
    yr = pd.to_datetime(d["touch_date"]).dt.year.to_numpy()

    results = []
    for fid, desc in FACTORS:
        if fid not in d.columns:
            continue
        v = d[fid].to_numpy(float)
        n_ok = int(np.isfinite(v).sum())
        if n_ok < 500:
            p(out, f"### {fid} — only {n_ok} rows with data, skipped\n")
            continue

        raw, mono, buckets = spread_of(v, y)
        c1m, c1s = perm_spread(v, y, sym)
        c2m, c2s = perm_spread(v, y, dt)
        worst = max(c1m, c2m) if (c1m == c1m and c2m == c2m) else np.nan
        net = raw - worst
        z1 = (raw - c1m) / c1s if c1s and c1s == c1s and c1s > 0 else np.nan
        z2 = (raw - c2m) / c2s if c2s and c2s == c2s and c2s > 0 else np.nan

        p(out, f"### {fid} — {desc}\n")
        p(out, f"| bucket | n | reversal rate |")
        p(out, f"|---:|---:|---:|")
        for k, n, r in buckets:
            p(out, f"| {k} | {n:,} | {r:.1f}% |")
        p(out, "")
        p(out, f"- raw spread        **{raw:+.2f}** points, monotonicity {mono:.2f}")
        p(out, f"- C1 within-stock   {c1m:+.2f} ± {c1s:.2f}   (z = {z1:+.1f})")
        p(out, f"- C2 within-date    {c2m:+.2f} ± {c2s:.2f}   (z = {z2:+.1f})")
        p(out, f"- **net of the worse control: {net:+.2f}**")

        # C3 — conditional on the two known survivors
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
            txt = "  ".join(f"{lab[i]} {g:+.1f}" if g == g else f"{lab[i]} n/a"
                            for i, g in enumerate(got))
            p(out, f"- C3 within {name} terciles: {txt}")

        # C4 — year by year
        ys = []
        for u in sorted(set(yr)):
            m = yr == u
            if m.sum() < 300:
                continue
            s_, _, _ = spread_of(v[m], y[m])
            ys.append((u, int(m.sum()), s_))
        if ys:
            p(out, "- C4 by year: " + "  ".join(f"{u} {s_:+.1f} (n={n})" for u, n, s_ in ys))
        agree = (all(s_ > 0 for _, _, s_ in ys) or all(s_ < 0 for _, _, s_ in ys)) if ys else False

        # pre-registered verdict, applied mechanically
        c3ok = sum(1 for g in cond.get("depth", []) if g == g and abs(g) >= BAR_NET / 2) >= 2
        passes = (net >= BAR_NET) and (mono == mono and mono >= BAR_MONO) and c3ok and agree
        verdict = "SURVIVES" if passes else "does not survive"
        why = []
        if not (net >= BAR_NET):
            why.append(f"net {net:+.1f} < {BAR_NET:+.0f}")
        if not (mono == mono and mono >= BAR_MONO):
            why.append(f"monotonicity {mono:.2f} < {BAR_MONO}")
        if not c3ok:
            why.append("dies inside depth bands")
        if not agree:
            why.append("sign flips across years")
        p(out, f"- **verdict: {verdict}**" + (f" — {', '.join(why)}" if why else ""))
        p(out, "")

        results.append(dict(side=side, factor=fid, n=n_ok, raw=raw, mono=mono,
                            c1=c1m, c2=c2m, net=net, verdict=verdict))
    return results


def p(out, s: str = "") -> None:
    print(s)
    out.append(s)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def synth(touches: pd.DataFrame, mode: str, seed: int = 3) -> pd.DataFrame:
    """Fabricate a delivery panel. `noise` has nothing in it — the harness must
    report nothing. `planted` carries a real +15-point effect on demand — the
    harness must find it. Neither is used for any published number."""
    rng = np.random.default_rng(seed)
    n = len(touches)
    base = rng.uniform(0.6, 1.6, n)
    if mode == "planted":
        lift = np.where(touches["side"].to_numpy() == "D",
                        (touches["label"].to_numpy() - 0.5) * 0.8, 0.0)
        base = base + lift + rng.normal(0, 0.35, n)
    out = touches[["touch_date", "nse_symbol"]].copy()
    out.columns = ["date", "symbol"]
    out["deliv_pct"] = np.clip(base * 45, 1, 100)
    out["deliv_pct_rel"] = base
    out["deliv_qty_rel"] = base * rng.uniform(0.8, 1.2, n)
    out["pre_deliv_pct_rel"] = rng.permutation(base)   # deliberately uninformative
    # a real delivery panel has one row per symbol-day; match that, or the merge
    # fans out on symbols touching two zones on the same date
    return out.drop_duplicates(subset=["date", "symbol"], keep="first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--touches", default="touches.csv")
    ap.add_argument("--delivery", default="data/nse_delivery.csv.gz")
    ap.add_argument("--selftest", choices=["noise", "planted"])
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    t = pd.read_csv(a.touches)
    if "nse_symbol" not in t.columns:
        t["nse_symbol"] = t["symbol"].str.replace(r"\.NS$", "", regex=True)

    if a.selftest:
        fac = synth(t, a.selftest)
        merged = t.merge(fac, left_on=["touch_date", "nse_symbol"],
                         right_on=["date", "symbol"], how="left", suffixes=("", "_f"))
        header = f"# SELF-TEST ({a.selftest}) — fabricated data, not a result"
    else:
        try:
            dlv = pd.read_csv(a.delivery)
        except FileNotFoundError:
            print(f"missing {a.delivery} — run fetch_delivery.py on a host that can "
                  f"reach NSE (the GitHub Action), then re-run this.", file=sys.stderr)
            return 2
        fac = build_factors(dlv)
        merged = t.merge(fac, left_on=["touch_date", "nse_symbol"],
                         right_on=["date", "symbol"], how="left", suffixes=("", "_f"))
        cov = merged["deliv_pct"].notna().mean() * 100
        header = (f"# Delivery percentage at zone touches\n\n"
                  f"{len(t):,} touches, {cov:.1f}% matched to a delivery record.\n"
                  f"Design fixed in advance — see PREREGISTRATION_delivery.md.")

    out: list[str] = []
    p(out, header)
    rows = []
    for side in ("D", "S"):
        rows += test_side(merged, side, out)

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
