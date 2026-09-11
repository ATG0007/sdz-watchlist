"""
fetch_oi.py — download NSE futures open interest and commit it beside the bars.

Same arrangement as fetch_delivery.py: Claude's sandbox cannot reach any NSE
host, so this runs on the GitHub Actions runner and writes data/nse_oi.csv.gz
for Claude to read over raw.githubusercontent.com.

Output columns: date, symbol, oi, oi_chg, fut_vol, n_expiries

Open interest is SUMMED ACROSS ALL FUTURES EXPIRIES for the symbol. Near-month
OI collapses at every monthly rollover, and that discontinuity would read as a
signal. Options are excluded — only stock futures.

Two file formats span the window and both are handled:
  * UDiFF   BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip   (mid-2024 onward)
  * legacy  fo<DD><MMM><YYYY>bhav.csv.zip                    (before that)

Usage
-----
    python fetch_oi.py                        # since 2023-09-01, incremental
    python fetch_oi.py --start 2024-01-01
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

HOME = "https://www.nseindia.com"
UDIFF = ("https://nsearchives.nseindia.com/content/fo/"
         "BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
LEGACY = ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
          "{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "text/csv,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{HOME}/all-reports-derivatives",
}

# instrument codes that mean "stock future" in each format
STOCK_FUT_LEGACY = {"FUTSTK"}
STOCK_FUT_UDIFF = {"STF"}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    for u in (HOME, f"{HOME}/all-reports-derivatives"):
        try:
            s.get(u, timeout=15)
        except requests.RequestException:
            pass
    return s


def _get(s: requests.Session, url: str, tries: int = 3) -> bytes | None:
    for k in range(tries):
        try:
            r = s.get(url, timeout=40)
            if r.status_code == 200 and r.content[:2] == b"PK":
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def _unzip(raw: bytes) -> pd.DataFrame | None:
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return None
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not names:
        return None
    df = pd.read_csv(z.open(names[0]), low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ""), errors="coerce")


def shape_udiff(df: pd.DataFrame) -> pd.DataFrame | None:
    need = {"FININSTRMTP", "TCKRSYMB", "OPNINTRST"}
    if not need.issubset(df.columns):
        return None
    d = df[df["FININSTRMTP"].astype(str).str.strip().str.upper().isin(STOCK_FUT_UDIFF)].copy()
    if d.empty:
        return None
    out = pd.DataFrame({
        "symbol": d["TCKRSYMB"].astype(str).str.strip(),
        "oi": _num(d["OPNINTRST"]),
        "oi_chg": _num(d["CHNGINOPNINTRST"]) if "CHNGINOPNINTRST" in d.columns else pd.NA,
        "fut_vol": _num(d["TTLTRADGVOL"]) if "TTLTRADGVOL" in d.columns else pd.NA,
    })
    return out


def shape_legacy(df: pd.DataFrame) -> pd.DataFrame | None:
    need = {"INSTRUMENT", "SYMBOL", "OPEN_INT"}
    if not need.issubset(df.columns):
        return None
    d = df[df["INSTRUMENT"].astype(str).str.strip().str.upper().isin(STOCK_FUT_LEGACY)].copy()
    if d.empty:
        return None
    out = pd.DataFrame({
        "symbol": d["SYMBOL"].astype(str).str.strip(),
        "oi": _num(d["OPEN_INT"]),
        "oi_chg": _num(d["CHG_IN_OI"]) if "CHG_IN_OI" in d.columns else pd.NA,
        "fut_vol": _num(d["CONTRACTS"]) if "CONTRACTS" in d.columns else pd.NA,
    })
    return out


def fetch_day(s: requests.Session, d: date) -> tuple[pd.DataFrame | None, str]:
    mon = d.strftime("%b").upper()
    for url, shaper, tag in (
        (UDIFF.format(d=d), shape_udiff, "udiff"),
        (LEGACY.format(d=d, mon=mon), shape_legacy, "legacy"),
    ):
        raw = _get(s, url)
        if raw is None:
            continue
        df = _unzip(raw)
        if df is None:
            continue
        try:
            got = shaper(df)
        except Exception as e:                                   # noqa: BLE001
            print(f"    parse error ({tag}): {e}", file=sys.stderr)
            got = None
        if got is None or got.empty:
            continue
        # one row per symbol: OI summed over every expiry
        agg = got.groupby("symbol", as_index=False).agg(
            oi=("oi", "sum"), oi_chg=("oi_chg", "sum"),
            fut_vol=("fut_vol", "sum"), n_expiries=("oi", "size"))
        agg.insert(0, "date", pd.Timestamp(d).date().isoformat())
        return agg, tag
    return None, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="data/nse_oi.csv.gz")
    ap.add_argument("--universe", default="universe.txt")
    ap.add_argument("--pause", type=float, default=0.8)
    a = ap.parse_args()

    keep = None
    if a.universe and os.path.exists(a.universe):
        keep = {ln.strip().upper() for ln in open(a.universe) if ln.strip()}
        print(f"universe: {len(keep)} symbols")

    start = pd.Timestamp(a.start).date()
    end = pd.Timestamp(a.end).date() if a.end else date.today()

    have, done = None, set()
    if os.path.exists(a.out):
        have = pd.read_csv(a.out)
        done = set(have["date"].astype(str))
        print(f"cache: {len(have):,} rows over {len(done)} dates")

    s = session()
    new, missing, fmt = [], 0, {"udiff": 0, "legacy": 0}
    d = start
    while d <= end:
        if d.weekday() < 5 and d.isoformat() not in done:
            got, tag = fetch_day(s, d)
            if got is None:
                missing += 1
            else:
                fmt[tag] = fmt.get(tag, 0) + 1
                if keep:
                    got = got[got["symbol"].str.upper().isin(keep)]
                new.append(got)
                print(f"  {d} {tag:6s} {len(got):4d} symbols", flush=True)
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
    print(f"rows     : {len(out):,}")
    print(f"symbols  : {out['symbol'].nunique()}")
    print(f"dates    : {out['date'].nunique()}  ({out['date'].min()} -> {out['date'].max()})")
    print(f"formats  : udiff {fmt.get('udiff', 0)} days, legacy {fmt.get('legacy', 0)} days")
    print(f"skipped  : {missing} weekday dates with no file (holidays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
