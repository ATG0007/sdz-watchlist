"""Fetch 3 years of daily bars for the study universe and write data/nse_daily.csv.gz.

Runs on a GitHub Actions runner, which has the internet access that Claude's
sandbox does not. Retries per batch, and fails loudly if the result looks wrong
rather than committing a half-empty file.
"""
import os, sys, time
import pandas as pd
import yfinance as yf

OUT = "data/nse_daily.csv.gz"
CHUNK, PAUSE, YEARS = 15, 1.5, 3
SYMBOLS = [s for s in open("universe.txt").read().split() if not s.startswith("#")]
TICKERS = [s + ".NS" for s in SYMBOLS] + ["^NSEI"]


def fetch(batch, attempt=1):
    try:
        d = yf.download(batch, period=f"{YEARS}y", progress=False, auto_adjust=False,
                        group_by="ticker", threads=False)
        if d is None or d.empty:
            raise ValueError("empty frame")
        d = d.stack(level=0, future_stack=True).reset_index()
        d.columns = [str(c) for c in d.columns]
        return d.rename(columns={d.columns[0]: "date", d.columns[1]: "symbol"})
    except Exception as exc:
        if attempt < 4:
            time.sleep(6 * attempt)
            return fetch(batch, attempt + 1)
        print(f"  batch failed after 4 tries: {exc}", flush=True)
        return None


def main():
    parts, failed = [], []
    batches = [TICKERS[i:i + CHUNK] for i in range(0, len(TICKERS), CHUNK)]
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

    n_sym, last = df.symbol.nunique(), pd.to_datetime(df.date).max().date()
    print(f"\n{len(df):,} rows | {n_sym} symbols | latest bar {last}")
    if failed:
        print(f"failed: {len(failed)} -> {', '.join(failed[:10])}")

    # refuse to commit a file that would produce a misleading watchlist
    if n_sym < len(TICKERS) * 0.8:
        sys.exit(f"FATAL: only {n_sym} of {len(TICKERS)} symbols returned data")
    if (pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timestamp(last)).days > 6:
        sys.exit(f"FATAL: latest bar is {last}, more than 6 days old")

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    with open("data/LATEST.txt", "w") as fh:
        fh.write(f"{last}\n{n_sym} symbols\n{len(df)} rows\n"
                 f"fetched {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
