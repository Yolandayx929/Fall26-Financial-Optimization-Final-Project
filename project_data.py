"""
Topic 3 data layer: download and clean CRSP, data-quality checks, universe summary.

Produces data/crsp_msf_v2.parquet, the file every notebook in analysis/ reads.
Universe selection itself is backtest.select_universe (the team standard); this module
only summarises the universe it picks.

Documented rules
----------------
Data source   crsp.msf_v2 (CIZ format, annual update), 1990-01 to 2025-12. In CIZ, mthret
              already includes the delisting return in a stock's final month.
Stocks        US common stocks, filtered in the SQL query: sharetype='NS',
              securitytype='EQTY', securitysubtype='COM', usincflg='Y',
              issuertype in ('ACOR','CORP'). The exchange filter (primaryexch in N, A, Q)
              is applied by the notebooks after loading.
Market cap    mthcap, in $ thousands (column mcap holds the same value in $ millions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import select_universe


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

COMMON_STOCK_SQL = """
SELECT permno, permco, yyyymm, mthcaldt, mthret, mthprc, shrout, mthcap,
       mthdelflg, primaryexch, siccd, ticker
FROM crsp.msf_v2
WHERE yyyymm BETWEEN {start} AND {end}
  AND sharetype = 'NS'
  AND securitytype = 'EQTY'
  AND securitysubtype = 'COM'
  AND usincflg = 'Y'
  AND issuertype IN ('ACOR', 'CORP')
"""


def download_crsp(db=None, start_yyyymm: int = 199001, end_yyyymm: int = 202512,
                  path: str | Path = "data/crsp_msf_v2.parquet",
                  refresh: bool = False) -> pd.DataFrame:
    """Load data/crsp_msf_v2.parquet, downloading it from WRDS first if it is missing.

    db : wrds.Connection, only needed when the file is missing or refresh=True.
    """
    path = Path(path)
    if path.exists() and not refresh:
        raw = pd.read_parquet(path)
        print(f"Loaded {path}: {raw.shape}")
        return raw

    if db is None:
        raise FileNotFoundError(f"{path} not found; pass a wrds.Connection to download it")
    raw = db.raw_sql(COMMON_STOCK_SQL.format(start=start_yyyymm, end=end_yyyymm),
                     date_cols=["mthcaldt"])
    raw = clean_crsp(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(path, index=False)
    print(f"Saved {path}: {raw.shape}")
    return raw


def clean_crsp(raw: pd.DataFrame) -> pd.DataFrame:
    """Type conversion, market cap in $ millions, one row per (permno, month)."""
    df = raw.copy()
    df["permno"] = df["permno"].astype("int64")
    df["permco"] = df["permco"].astype("int64")
    df["yyyymm"] = df["yyyymm"].astype("int64")
    for c in ["mthret", "mthprc", "shrout", "mthcap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # mthcap is in $ thousands; fall back to |price| * shares (shares in thousands)
    cap = df["mthcap"] / 1000.0
    fallback = df["mthprc"].abs() * df["shrout"] / 1000.0
    df["mcap"] = cap.fillna(fallback)
    df.loc[df["mcap"] <= 0, "mcap"] = np.nan

    df["month"] = pd.PeriodIndex(df["yyyymm"].astype(str), freq="M")
    df = (df.sort_values(["permno", "month", "mcap"])
            .drop_duplicates(["permno", "month"], keep="last")
            .reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# 2. Data-quality checks and universe summary
# ---------------------------------------------------------------------------

def data_quality_report(raw: pd.DataFrame) -> pd.Series:
    r = raw["mthret"]
    return pd.Series({
        "rows": len(raw),
        "unique permnos": raw["permno"].nunique(),
        "first month": int(raw["yyyymm"].min()),
        "last month": int(raw["yyyymm"].max()),
        "duplicate (permno, month)": int(raw.duplicated(["permno", "yyyymm"]).sum()),
        "missing mthret (%)": round(100 * r.isna().mean(), 3),
        "missing mthcap (%)": round(100 * raw["mthcap"].isna().mean(), 3),
        "returns < -100%": int((r < -1).sum()),
        "returns > +100%": int((r > 1).sum()),
        "returns > +300%": int((r > 3).sum()),
        "min return": r.min(),
        "max return": r.max(),
        "delisting months (mthdelflg != 'N')": int((raw["mthdelflg"] != "N").sum()),
        "not on NYSE/AMEX/NASDAQ (%)": round(100 * (~raw["primaryexch"].isin(["N", "A", "Q"])).mean(), 3),
    }, dtype=object)   # object dtype keeps counts as integers when displayed


def universe_summary(rets: pd.DataFrame, caps: pd.DataFrame, T: int, n: int,
                     start, end) -> pd.DataFrame:
    """One row per rebalance date t in [start, end], for the universe select_universe picks.

    rets, caps : month x permno panels as used by backtest.rolling_backtest
                 (caps in $ thousands, reported here in $ millions).
    """
    idx = rets.index
    rows, prev = [], None
    for i in range(idx.get_loc(start), idx.get_loc(end) + 1):
        t, t_next = idx[i], idx[i + 1]
        window = idx[i - T + 1: i + 1]
        universe = select_universe(rets, caps, window, t, n)

        full_history = rets.loc[window].notna().all(axis=0)
        cap_t = caps.loc[t]
        n_eligible = int((full_history & (cap_t > 0)).sum())
        cap_u = cap_t[universe] / 1e3
        cur = set(universe)
        rows.append(dict(
            rebalance=t,
            n_eligible=n_eligible,
            n_universe=len(universe),
            min_cap_musd=cap_u.min(),
            median_cap_musd=cap_u.median(),
            share_of_total_cap=cap_u.sum() / (cap_t.sum() / 1e3),
            top10_weight_vw=cap_u.nlargest(10).sum() / cap_u.sum(),
            new_names=np.nan if prev is None else len(cur - prev),
            missing_next_ret=int(rets.loc[t_next, universe].isna().sum()),
        ))
        prev = cur
    return pd.DataFrame(rows).set_index("rebalance")
