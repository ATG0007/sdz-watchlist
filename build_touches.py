"""
build_touches.py — rebuild the zone-touch dataset from the repo's daily bars.

Emits touches.csv with the sdz_core schema plus `nse_symbol` (the bare NSE
ticker) so a delivery-percentage table can be joined on (nse_symbol, touch_date).
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import sdz_core


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/nse_daily.csv.gz")
    ap.add_argument("--out", default="touches.csv")
    a = ap.parse_args()

    px = pd.read_csv(a.prices)
    px.columns = [c.strip().lower().replace(" ", "_") for c in px.columns]
    # unadjusted, to match the watchlist and the chart
    px = px.rename(columns={"adj_close": "_adj"})
    px["date"] = pd.to_datetime(px["date"])

    rows = []
    symbols = sorted(px["symbol"].unique())
    for n, sym in enumerate(symbols, 1):
        s = px[px["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(s) < 200:
            continue
        try:
            got = sdz_core.extract_touches(s[["date", "open", "high", "low", "close", "volume"]], sym)
        except Exception as e:                      # noqa: BLE001
            print(f"  ! {sym}: {e}", file=sys.stderr)
            continue
        rows.extend(got)
        if n % 40 == 0:
            print(f"  {n}/{len(symbols)} symbols, {len(rows)} touches", file=sys.stderr)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no touches extracted", file=sys.stderr)
        return 1
    df["nse_symbol"] = df["symbol"].str.replace(r"\.NS$", "", regex=True)
    df.to_csv(a.out, index=False)

    print(f"touches      : {len(df):,}")
    print(f"symbols      : {df['symbol'].nunique()}")
    print(f"date range   : {df['touch_date'].min()} -> {df['touch_date'].max()}")
    print(f"demand/supply: {(df['side'] == 'D').sum():,} / {(df['side'] == 'S').sum():,}")
    print(f"base rate    : {df['label'].mean() * 100:.2f}%  "
          f"(D {df.loc[df.side == 'D', 'label'].mean() * 100:.2f}%, "
          f"S {df.loc[df.side == 'S', 'label'].mean() * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
