# SDZ daily watchlist

Every weekday evening a GitHub Action downloads 3 years of daily bars for ~198
NSE names and commits them here. Claude then reads that file and sends the top
25 stocks sitting near a live demand zone.

The Action exists because Claude's sandbox has no access to Yahoo, NSE or any
market-data host — only PyPI and GitHub. GitHub's runners have the internet that
Claude does not, so they do the fetching and Claude reads the result.

## One-time setup

1. Create a **new public repository** on GitHub — name it `sdz-watchlist`.
   Public matters: `raw.githubusercontent.com` serves public files without a
   token. There is nothing private here, only public market prices.
2. Upload every file from this folder, keeping the structure:
   ```
   .github/workflows/fetch.yml
   fetch_nse.py
   sdz_core.py
   sdz_scan.py
   sdz_top25.py
   universe.txt
   ```
3. Open the **Actions** tab and enable workflows if GitHub asks.
4. Click **Fetch NSE daily bars** -> **Run workflow** to test it now. It takes
   about 5 minutes. When it finishes, `data/nse_daily.csv.gz` should exist.
5. Tell Claude your GitHub username and repo name.

## Schedule

| When (IST) | What |
|---|---|
| 16:10 Mon–Fri | Action downloads and commits |
| 16:35 Mon–Fri | Second attempt, in case Yahoo's bar was late |
| 16:50 Sun–Thu | Claude reads the file and sends the top 25 |

Sunday's run uses Friday's close, which is correct for planning Monday.

## Running it yourself

```bash
pip install yfinance pandas
python fetch_nse.py
python sdz_top25.py
```

## What the ranking is and is not

Sorted by **stop distance**, then by how close price is to the zone. It is
**not** sorted by any probability of success. A 20-year study over 77,927 zone
touches found that none of the candidate quality measures — Zone Score, Hold
Score, counter-trend context, touch number, zone age — beat a random-data null
model. Distance and stop size are ranked because they are the two things that
genuinely change the outcome: they set how much you can size and how far price
must travel.

Levels are **not dividend-adjusted**, so they match a TradingView chart.
