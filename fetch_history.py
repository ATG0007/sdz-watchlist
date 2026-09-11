"""Fetch 20 years of daily bars for a broad NSE universe, for the STUDY only.

Separate from fetch_nse.py by design. The watchlist keeps running off
data/nse_daily.csv.gz (3 years, F&O list); this writes shards under data/hist/
and nothing downstream of the daily scan touches them. Run once from the
Actions tab - there is no schedule, because study data does not change.

UNIVERSE
--------
Pulls the Nifty 500 constituent list from NSE at run time, so the universe is
not a hand-typed list that rots. Falls back to universe.txt if NSE refuses.

Nifty 500 rather than "today's top 400" on purpose. Both are survivorship-
tilted, but the 500 reaches far enough down that it still contains names which
FELL out of the top tier and kept trading - the IDEAs, YESBANKs, ZEELs,
RBLBANKs. Those are exactly the stocks whose demand zones failed, and a study
that excludes them will overstate every reversal rate it measures. Companies
that delisted entirely are still missing; Yahoo does not carry them, and no
free source does.

SHARDS
------
Written as 4 gzipped parts, each small enough to attach to a chat. A single
20-year file for 500 names lands near GitHub's size warning and is awkward to
move by hand.
"""
import hashlib
import io
import os
import sys
import time

import pandas as pd
import requests
import yfinance as yf

OUTDIR = "data/hist"
NPARTS = 4
CHUNK, PAUSE, YEARS = 10, 2.0, 20
NIFTY500 = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")

# Yahoo needs a different ticker for these, or the name is a duplicate/dead
FIX = {"BERGERPAINT": None,      # duplicate of BERGEPAINT
       "IDFC": None,             # merged into IDFCFIRSTB
       "ZOMATO": "ETERNAL",      # renamed
       "LTIM": "LTIM",           # kept; Yahoo may or may not serve it
       "M&M": "M&M", "M&MFIN": "M&MFIN"}


def universe() -> list[str]:
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept": "text/csv,*/*"})
        s.get("https://www.nseindia.com", timeout=15)
        r = s.get(NIFTY500, timeout=30)
        if r.status_code == 200 and b"Symbol" in r.content[:400]:
            df = pd.read_csv(io.BytesIO(r.content))
            col = [c for c in df.columns if c.strip().lower() == "symbol"][0]
            syms = sorted({str(x).strip().upper() for x in df[col] if str(x).strip()})
            print(f"universe: Nifty 500 from NSE - {len(syms)} symbols")
            return syms
        print(f"  NSE list returned {r.status_code}; falling back", flush=True)
    except requests.RequestException as e:
        print(f"  NSE list failed ({e}); falling back", flush=True)
    syms = sorted({s.strip().upper() for s in open("universe.txt").read().split()
                   if s.strip() and not s.startswith("#")})
    print(f"universe: universe.txt fallback - {len(syms)} symbols")
    return syms


def fetch(batch, attempt=1):
    try:
        d = yf.download(batch, period=f"{YEARS}y", progress=False, auto_adjust=False,
                        group_by="ticker", threads=False)
        if d is None or d.empty:
            raise ValueError("empty frame")
        d = d.stack(level=0, future_stack=True).reset_index()
        d.columns = [str(c) for c in d.columns]
        return d.rename(columns={d.columns[0]: "date", d.columns[1]: "symbol"})
    except Exception as exc:                                    # noqa: BLE001
        if attempt < 4:
            time.sleep(8 * attempt)
            return fetch(batch, attempt + 1)
        print(f"  batch failed after 4 tries: {exc}", flush=True)
        return None


def main():
    syms = universe()
    syms = [FIX.get(s, s) for s in syms]
    syms = sorted({s for s in syms if s})
    tickers = [s + ".NS" for s in syms] + ["^NSEI"]
    print(f"requesting {len(tickers)} tickers, {YEARS} years each\n")

    parts, failed = [], []
    batches = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    for bi, b in enumerate(batches, 1):
        d = fetch(b)
        if d is not None and len(d):
            parts.append(d)
            print(f"[{bi}/{len(batches)}] {len(d):,} rows", flush=True)
        else:
            failed.extend(b)
        time.sleep(PAUSE)

    if not parts:
        sys.exit("FATAL: nothing downloaded")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["Close"])
    df = df.drop_duplicates(subset=["date", "symbol"]).sort_values(["symbol", "date"])
    keep = ["date", "symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])

    n_sym = df.symbol.nunique()
    first, last = df.date.min().date(), df.date.max().date()
    print(f"\n{len(df):,} rows | {n_sym} symbols | {first} -> {last}")

    # --- the table that belongs on the front of any finding -----------------
    per = df.groupby("symbol")["date"].agg(["min", "max", "size"])
    per["years"] = (per["max"] - per["min"]).dt.days / 365.25
    print("\nHISTORY DEPTH - a 20-year request is not 20 years of data")
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 25)):
        n = int(((per.years >= lo) & (per.years < hi)).sum())
        bars = int(per.loc[(per.years >= lo) & (per.years < hi), "size"].sum())
        print(f"  {lo:>2}-{hi:<2} years : {n:>4} symbols, {bars:>9,} bars")
    print(f"  median history: {per.years.median():.1f} years")
    print(f"  symbols listed after 2016: "
          f"{int((per['min'] > pd.Timestamp('2016-01-01')).sum())}")

    if failed:
        print(f"\nfailed: {len(failed)} -> {', '.join(failed[:15])}")
    if n_sym < 250:
        sys.exit(f"FATAL: only {n_sym} symbols returned data")

    os.makedirs(OUTDIR, exist_ok=True)
    df["_part"] = [int(hashlib.md5(s.encode()).hexdigest(), 16) % NPARTS
                   for s in df["symbol"]]
    for k in range(NPARTS):
        sh = df[df._part == k].drop(columns="_part")
        path = f"{OUTDIR}/nse_20y_part{k + 1}.csv.gz"
        sh.to_csv(path, index=False, compression="gzip")
        print(f"wrote {path}  {len(sh):,} rows, {sh.symbol.nunique()} symbols, "
              f"{os.path.getsize(path) / 1e6:.1f} MB")

    with open("data/LATEST_20Y.txt", "w") as fh:
        fh.write(f"{first} -> {last}\n{n_sym} symbols\n{len(df)} rows\n"
                 f"median history {per.years.median():.1f} years\n"
                 f"{NPARTS} shards under {OUTDIR}/\n"
                 f"fetched {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC\n")


if __name__ == "__main__":
    main()
