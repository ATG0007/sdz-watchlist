"""
fetch_delivery.py — download NSE delivery data and commit it beside the bars.

Claude's sandbox cannot reach any NSE host (the egress proxy denies it), exactly
as it cannot reach Yahoo. This runs where fetch_nse.py already runs — on a
GitHub Actions runner — and writes data/nse_delivery.csv.gz for Claude to read
over raw.githubusercontent.com.

Output columns: date, symbol, ttl_qty, deliv_qty, deliv_pct

Sources, tried in this order for each date:
  1. sec_bhavdata_full_DDMMYYYY.csv   — the full security-wise report
  2. MTO_DDMMYYYY.DAT                 — the older "securities to market" file

Both are public. Neither needs a login. NSE rejects requests without a browser
UA and a cookie from the main site, so the session warms up first.

Usage
-----
    python fetch_delivery.py                       # since 2023-09-01, incremental
    python fetch_delivery.py --start 2024-01-01
    python fetch_delivery.py --universe universe.txt   # keep only these symbols
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

HOME = "https://www.nseindia.com"
FULL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"
MTO = "https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d:%d%m%Y}.DAT"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{HOME}/all-reports",
}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(HOME, timeout=15)
        s.get(f"{HOME}/all-reports", timeout=15)
    except requests.RequestException as e:                      # noqa: BLE001
        print(f"  warm-up failed ({e}) — continuing anyway", file=sys.stderr)
    return s


def _get(s: requests.Session, url: str, tries: int = 3) -> bytes | None:
    for k in range(tries):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200 and r.content and b"<html" not in r.content[:200].lower():
                return r.content
            if r.status_code == 404:
                return None                     # holiday or not published
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def parse_full(raw: bytes) -> pd.DataFrame | None:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().upper() for c in df.columns]
    need = {"SYMBOL", "SERIES", "DELIV_QTY", "DELIV_PER", "TTL_TRD_QNTY"}
    if not need.issubset(df.columns):
        return None
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    for c in ("DELIV_QTY", "DELIV_PER", "TTL_TRD_QNTY"):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["DELIV_PER"])
    return pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "ttl_qty": df["TTL_TRD_QNTY"],
        "deliv_qty": df["DELIV_QTY"],
        "deliv_pct": df["DELIV_PER"],
    })


def parse_mto(raw: bytes) -> pd.DataFrame | None:
    """MTO_*.DAT — header lines, then: recno,symbol,series,traded,delivered,pct."""
    rows = []
    for line in raw.decode("utf-8", "ignore").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or parts[2] != "EQ":
            continue
        try:
            rows.append((parts[1], float(parts[3]), float(parts[4]), float(parts[5])))
        except ValueError:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["symbol", "ttl_qty", "deliv_qty", "deliv_pct"])


def fetch_day(s: requests.Session, d: date) -> pd.DataFrame | None:
    for url, parser in ((FULL.format(d=d), parse_full), (MTO.format(d=d), parse_mto)):
        raw = _get(s, url)
        if raw is None:
            continue
        try:
            got = parser(raw)
        except Exception:                                        # noqa: BLE001
            got = None
        if got is not None and len(got):
            got.insert(0, "date", pd.Timestamp(d).date().isoformat())
            return got
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/nse_delivery.csv.gz")
    ap.add_argument("--universe", default="universe.txt")
    ap.add_argument("--pause", type=float, default=0.8)
    a = ap.parse_args()

    keep = None
    if a.universe and os.path.exists(a.universe):
        keep = {ln.strip().upper() for ln in open(a.universe) if ln.strip()}
        print(f"universe: {len(keep)} symbols")

    start = pd.Timestamp(a.start).date()
    end = pd.Timestamp(a.end).date() if a.end else date.today()

    have: pd.DataFrame | None = None
    done: set[str] = set()
    if os.path.exists(a.out):
        have = pd.read_csv(a.out)
        done = set(have["date"].astype(str))
        print(f"cache: {len(have):,} rows over {len(done)} dates — fetching only what is missing")

    s = session()
    new, missing, d = [], 0, start
    while d <= end:
        if d.weekday() < 5 and d.isoformat() not in done:
            got = fetch_day(s, d)
            if got is None:
                missing += 1                     # holiday, or not published yet
            else:
                if keep:
                    got = got[got["symbol"].str.upper().isin(keep)]
                new.append(got)
                print(f"  {d} {len(got):5d} rows", flush=True)
            time.sleep(a.pause)
        d += timedelta(days=1)

    if not new and have is None:
        print("nothing fetched — NSE unreachable from here?", file=sys.stderr)
        return 1

    out = pd.concat(([have] if have is not None else []) + new, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
    out = out.sort_values(["symbol", "date"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False, compression="gzip")

    print(f"\nwrote {a.out}")
    print(f"rows    : {len(out):,}")
    print(f"symbols : {out['symbol'].nunique()}")
    print(f"dates   : {out['date'].nunique()}  ({out['date'].min()} -> {out['date'].max()})")
    print(f"skipped : {missing} weekday dates with no file (holidays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
