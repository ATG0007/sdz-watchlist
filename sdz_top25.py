"""sdz_top25.py — the evening watchlist.

Reads data/nse_daily.csv.gz, finds every live demand zone, and prints the 25
best candidates plus a freshness check. Ranked by how close price is to the zone
and how tight the stop would be - not by any "probability", because the 20-year
study found no filter that beat a null model.
"""
import argparse, sys
import numpy as np, pandas as pd
from sdz_core import prepare
from sdz_scan import live_demand_zones

ap = argparse.ArgumentParser()
ap.add_argument("--prices", default="data/nse_daily.csv.gz")
ap.add_argument("--max-atr", type=float, default=1.2)
ap.add_argument("--max-risk", type=float, default=5.0, help="max stop distance, % of entry")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--capital", type=float, default=10000.0)
ap.add_argument("--out", default="watchlist.csv")
a = ap.parse_args()

raw = pd.read_csv(a.prices)
raw.columns = [c.strip().lower() for c in raw.columns]
raw = raw.rename(columns={"adj close": "adj_close"})
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw = raw.dropna(subset=["date"]).drop_duplicates(subset=["date", "symbol"])
last = raw["date"].max()
age_days = (pd.Timestamp.now("UTC").tz_localize(None).normalize() - last.normalize()).days

rows = []
for sym, g in raw.groupby("symbol"):
    if str(sym).startswith("^"):
        continue
    name = str(sym).replace(".NS", "")
    g = g.sort_values("date")
    if g["date"].max() < last - pd.Timedelta(days=5) or len(g) < 300:
        continue
    d = prepare(g)                       # NOTE: no dividend adjustment - matches TradingView
    if len(d) < 300:
        continue
    zones, n = live_demand_zones(d)
    i = n - 1
    close, atr = d.close.values[i], d.atr.values[i]
    if not np.isfinite(atr) or atr <= 0:
        continue
    for z in zones:
        gap = (close - z["top"]) / atr
        if gap > a.max_atr or close < z["bot"]:
            continue
        stop = z["bot"] - 0.25 * atr
        risk = z["top"] - stop
        rp = risk / z["top"] * 100
        if rp > a.max_risk:
            continue
        rows.append(dict(
            symbol=name, status="INSIDE" if gap <= 0 else "approaching",
            close=round(close, 2), buy_at=round(z["top"], 2), zone_bot=round(z["bot"], 2),
            stop=round(stop, 2), risk_pct=round(rp, 2),
            target_2R=round(z["top"] + 2 * risk, 2),
            gain_pct=round(2 * risk / z["top"] * 100, 2),
            shares=int(a.capital / risk) if risk > 0 else 0,
            gap_atr=round(gap, 2), width_atr=round((z["top"] - z["bot"]) / atr, 2),
            age_bars=int(i - z["left"]), touches=int(z["touches"])))

df = pd.DataFrame(rows)
if df.empty:
    print(f"DATA_DATE={last.date()} AGE_DAYS={age_days} ROWS=0")
    sys.exit("no live demand zones within range")

df = df.sort_values(["risk_pct", "gap_atr"]).drop_duplicates("symbol").head(a.top)
df = df.reset_index(drop=True)
df.to_csv(a.out, index=False)

print(f"DATA_DATE={last.date()}  AGE_DAYS={age_days}  SYMBOLS={raw.symbol.nunique()}  SHOWN={len(df)}")
print("STALE - the newest bar is more than 4 days old" if age_days > 4 else "FRESH")
print()
print(f"{'#':<4}{'stock':<13}{'status':<13}{'close':>10}{'buy at':>10}{'stop':>10}"
      f"{'risk':>7}{'target':>10}{'gain':>7}{'sh':>7}{'age':>5}{'t':>3}")
print("-" * 108)
for r in df.itertuples():
    print(f"{r.Index+1:<4}{r.symbol:<13}{r.status:<13}{r.close:>10.2f}{r.buy_at:>10.2f}"
          f"{r.stop:>10.2f}{r.risk_pct:>6.1f}%{r.target_2R:>10.2f}{r.gain_pct:>6.1f}%"
          f"{r.shares:>7}{r.age_bars:>5}{r.touches:>3}")
print("-" * 108)
print("risk = entry to stop, % of entry. sh = shares for Rs 10,000 risked.")
print("age = bars since the zone formed. t = times already tested.")
