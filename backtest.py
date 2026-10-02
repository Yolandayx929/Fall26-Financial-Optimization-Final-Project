"""Rolling out-of-sample backtest.

At every rebalance date t the procedure is:

  1. select the universe using only information known at t,
  2. estimate the covariance model on the previous T months,
  3. build each portfolio,
  4. score it on the return of month t+1.

Expected inputs (monthly, in decimals, not percentage points):

  rets : DataFrame, index = month-end dates ascending, columns = permno,
         values = total return
  caps : DataFrame, same shape as rets, values = market capitalisation
  mkt  : Series indexed like rets, market excess return
  rf   : Series indexed like rets, risk-free rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estimation import estimate_single_factor
from portfolios import min_risk_closed_form

# The closed form is ~3900x faster than the cvxpy model on n=500 (0.2ms vs 776ms),
# which turns a 372-month backtest from ~5 minutes into a fraction of a second.
# `min_risk` from portfolios.py solves the same problem and is kept for
# cross-validation.
CONSTRUCTORS = {"min_risk": min_risk_closed_form}


def select_universe(rets, caps, window, t, n):
    """Largest n stocks by market cap at t, with a complete return history.

    Both filters use only data available at t, so there is no look-ahead.
    Requiring a complete window does tilt the universe towards longer-listed
    stocks; this is a known and reported limitation.
    """
    history = rets.loc[window]
    eligible = history.columns[history.notna().all(axis=0)]
    cap_t = caps.loc[t, eligible].dropna()
    cap_t = cap_t[cap_t > 0]
    return cap_t.nlargest(min(n, len(cap_t))).index


def rolling_backtest(
    rets,
    caps,
    mkt,
    rf,
    T: int = 60,
    n: int = 500,
    start=None,
    end=None,
    constructors: dict | None = None,
):
    """Run the rolling backtest.

    Returns
    -------
    panel : DataFrame indexed by (date, strategy), one row per month per
        strategy, where `date` is the month the return was earned.
    weights : dict mapping strategy name to a DataFrame of weights
        (index = rebalance date, columns = permno).
    """
    constructors = constructors or CONSTRUCTORS
    index = rets.index

    first = T - 1 if start is None else max(T - 1, index.get_loc(start))
    last = len(index) - 2 if end is None else min(len(index) - 2, index.get_loc(end))

    records = []
    weights = {name: {} for name in constructors}
    previous = {name: None for name in constructors}

    for i in range(first, last + 1):
        t, t_next = index[i], index[i + 1]
        window = index[i - T + 1 : i + 1]

        universe = select_universe(rets, caps, window, t, n)
        if len(universe) < 20:
            continue

        excess = rets.loc[window, universe].values - rf.loc[window].values[:, None]
        cov = estimate_single_factor(
            excess, mkt.loc[window].values, permno=universe.values
        )

        # Out-of-sample month. If a holding has no return at t+1 its weight is
        # assumed to earn the risk-free rate. Renormalising over the survivors
        # instead would amount to knowing at the start of the month which
        # stocks were about to disappear.
        r_next = rets.loc[t_next, universe].fillna(float(rf.loc[t_next])).values

        for name, construct in constructors.items():
            # caps are passed through for constructors that need them; those that
            # do not simply absorb the keyword.
            result = construct(cov, caps=caps.loc[t, universe].values)
            x = result["x"]
            r_portfolio = float(x @ r_next)

            if previous[name] is None:
                turnover = np.nan  # first period has no predecessor
            else:
                current = pd.Series(x, index=universe)
                turnover = 0.5 * float(
                    current.sub(previous[name], fill_value=0.0).abs().sum()
                )

            # Turnover is measured against last month's drifted weights, not
            # against the weights as originally set.
            previous[name] = pd.Series(
                x * (1.0 + r_next) / (1.0 + r_portfolio), index=universe
            )
            weights[name][t] = pd.Series(x, index=universe)

            records.append(
                dict(
                    date=t_next,
                    strategy=name,
                    ret=r_portfolio,
                    exret=r_portfolio - float(rf.loc[t_next]),
                    pred_vol=float(np.sqrt(max(result["sigma2"], 0.0))),
                    exante_beta=cov.exante_beta(x),
                    n_held=result["n_held"],
                    # only the closed form exposes the holding threshold
                    beta_LO=result.get("beta_LO", np.nan),
                    eff_n=float(1.0 / np.sum(x**2)),
                    max_w=float(x.max()),
                    turnover=turnover,
                )
            )

    panel = pd.DataFrame(records).set_index(["date", "strategy"]).sort_index()
    return panel, {name: pd.DataFrame(w).T for name, w in weights.items()}


def summarize(panel, periods_per_year: int = 12) -> pd.DataFrame:
    """Annualised summary statistics, one row per strategy."""
    rows = {}
    for name, group in panel.groupby(level="strategy"):
        excess = group["exret"].values
        total = group["ret"].values
        vol = excess.std(ddof=1) * np.sqrt(periods_per_year)
        cumulative = np.cumprod(1.0 + total)
        drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0
        rows[name] = dict(
            ann_return=cumulative[-1] ** (periods_per_year / len(total)) - 1.0,
            ann_vol=vol,
            sharpe=excess.mean() * periods_per_year / vol,
            max_drawdown=drawdown.min(),
            pred_over_realized=(
                group["pred_vol"].mean() * np.sqrt(periods_per_year) / vol
            ),
            avg_n_held=group["n_held"].mean(),
            avg_eff_n=group["eff_n"].mean(),
            ann_turnover=group["turnover"].mean() * periods_per_year,
        )
    return pd.DataFrame(rows).T
