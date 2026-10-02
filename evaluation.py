"""Backtesting analysis, applied identically to every strategy.

Covers performance metrics, turnover, risk decomposition and concentration
diagnostics, and the cross-strategy alignment checks.

Conventions, chosen to match `backtest.summarize` exactly so the two can be
cross-checked:

  * returns are monthly decimals, indexed by the month they were earned (yyyymm)
  * annualised return is geometric
  * annualised volatility and the Sharpe ratio use *excess* returns
  * maximum drawdown is taken on the compounded total-return path
  * turnover is one-way, 0.5 * sum |w_t - w_{t-1}^drifted|, measured against
    the previous weights *after* they drifted with realised returns, so that
    passive drift is not counted as trading; the first rebalance has none
  * a holding with no return in a month earns the risk-free rate

Weights histories are DataFrames indexed by rebalance date t, with one column
per permno and NaN for names outside that date's universe -- the format that
`backtest.rolling_backtest` returns. A weight set at t earns the return of the
next month in the return panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estimation import FactorCov, estimate_single_factor


# --------------------------------------------------------------------------
# Return series
# --------------------------------------------------------------------------

def returns_from_panel(panel, column: str = "ret") -> pd.DataFrame:
    """Reshape a `rolling_backtest` panel into a month x strategy frame."""
    return panel[column].unstack("strategy")


def cumulative_returns(returns) -> pd.DataFrame | pd.Series:
    """Growth of $1 minus one, compounded month by month."""
    return (1.0 + returns).cumprod() - 1.0


def drawdowns(returns) -> pd.DataFrame | pd.Series:
    """Drawdown from the running peak of the compounded path."""
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1.0


# --------------------------------------------------------------------------
# Turnover
# --------------------------------------------------------------------------

def drifted_turnover(weights_history: pd.DataFrame, asset_returns: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """One-way turnover at each rebalance date, against drifted weights.

    Between consecutive rebalances t_{k-1} and t_k the previous weights grow
    with every month's realised return in (t_{k-1}, t_k] and are renormalised.
    For monthly rebalancing that is a single month, which reproduces the
    turnover recorded by `rolling_backtest`; less frequent rebalancing is
    handled by compounding over the intervening months.

    Returns a Series indexed by rebalance date; the first entry is NaN.
    """
    months = asset_returns.index
    dates = weights_history.index
    out = pd.Series(np.nan, index=dates, dtype=float)

    for prev_t, t in zip(dates[:-1], dates[1:]):
        prev = weights_history.loc[prev_t].dropna()
        held = months[months.get_loc(prev_t) + 1 : months.get_loc(t) + 1]
        growth = np.ones(len(prev))
        for m in held:
            r = asset_returns.loc[m, prev.index].fillna(float(rf.loc[m])).values
            growth *= 1.0 + r
        drifted = prev * growth
        drifted /= drifted.sum()

        current = weights_history.loc[t].dropna()
        out[t] = 0.5 * float(current.sub(drifted, fill_value=0.0).abs().sum())
    return out


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def performance_metrics(
    return_series: pd.Series,
    weights_history: pd.DataFrame | None = None,
    *,
    rf: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    mkt: pd.Series | None = None,
    periods_per_year: int = 12,
) -> pd.Series:
    """Summary statistics for one strategy.

    Parameters
    ----------
    return_series : monthly total returns, indexed by the month earned.
        Passing a slice evaluates a subperiod; turnover is then restricted to
        the rebalances whose holding month falls inside that slice.
    weights_history : rebalance date x permno weights. Needed for turnover.
    rf : risk-free rate, indexed like the return panel.
    asset_returns : the month x permno return panel. Needed for turnover.
    mkt : market excess return. If given, the realised beta is reported.

    Returns
    -------
    Series with months, ann_return, ann_vol, sharpe, max_drawdown,
    cum_return, ann_turnover (and beta when `mkt` is given).
    """
    r = return_series.dropna()
    if len(r) < 2:
        raise ValueError("need at least two months of returns")
    excess = r - rf.reindex(r.index)
    if excess.isna().any():
        raise ValueError("risk-free rate missing for some months")

    wealth = np.cumprod(1.0 + r.values)
    vol = excess.std(ddof=1) * np.sqrt(periods_per_year)
    stats = dict(
        months=len(r),
        ann_return=wealth[-1] ** (periods_per_year / len(r)) - 1.0,
        ann_vol=vol,
        sharpe=excess.mean() * periods_per_year / vol,
        max_drawdown=float((wealth / np.maximum(1.0, np.maximum.accumulate(wealth)) - 1.0).min()),
        cum_return=wealth[-1] - 1.0,
        ann_turnover=np.nan,
    )

    if weights_history is not None and asset_returns is not None:
        turnover = drifted_turnover(weights_history, asset_returns, rf)
        months = asset_returns.index
        held_month = [months[months.get_loc(t) + 1] for t in turnover.index]
        turnover = turnover[pd.Index(held_month).isin(r.index)]
        stats["ann_turnover"] = turnover.mean() * periods_per_year

    if mkt is not None:
        m = mkt.reindex(r.index)
        stats["beta"] = float(np.cov(excess, m, ddof=1)[0, 1] / m.var(ddof=1))

    return pd.Series(stats, name=return_series.name)


def performance_table(
    returns: pd.DataFrame,
    weights: dict | None = None,
    *,
    rf: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    mkt: pd.Series | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """`performance_metrics` for every strategy, one row each.

    `returns` is month x strategy; `weights` maps strategy name to its weights
    history. `start` / `end` (inclusive, yyyymm) select a subperiod.
    """
    window = returns.loc[start:end]
    rows = [
        performance_metrics(
            window[name],
            None if weights is None else weights.get(name),
            rf=rf,
            asset_returns=asset_returns,
            mkt=mkt,
        ).rename(name)
        for name in window.columns
    ]
    return pd.DataFrame(rows)


def subperiod_table(returns, weights=None, *, periods, **kwargs) -> pd.DataFrame:
    """`performance_table` over several (start, end, label) periods, stacked."""
    return pd.concat(
        {label: performance_table(returns, weights, start=lo, end=hi, **kwargs) for lo, hi, label in periods},
        names=["period", "strategy"],
    )


# --------------------------------------------------------------------------
# Risk decomposition and concentration diagnostics
# --------------------------------------------------------------------------

def _variance_gradient(x, cov) -> np.ndarray:
    """Vx, for either a FactorCov or a dense matrix."""
    x = np.asarray(x, dtype=float)
    if isinstance(cov, FactorCov):
        return cov.sigmaM2 * cov.beta * float(cov.beta @ x) + cov.omega2 * x
    return np.asarray(cov, dtype=float) @ x


def risk_contributions(x, cov) -> dict:
    """Euler decomposition of portfolio volatility.

    sigma_p = sqrt(x'Vx) is homogeneous of degree one in x, so Euler's theorem
    splits it exactly across the holdings:

        MRC_i = d sigma_p / d x_i = (Vx)_i / sigma_p        marginal contribution
        RC_i  = x_i * MRC_i                                 total contribution
        sum_i RC_i = sigma_p

    Parameters
    ----------
    x : weights, shape (n,).
    cov : FactorCov or dense covariance matrix of shape (n, n).

    Returns
    -------
    dict with
      sigma_p : portfolio volatility sqrt(x'Vx)
      mrc     : marginal risk contributions (Vx)_i / sigma_p
      rc      : total risk contributions x_i * mrc_i, summing to sigma_p
      share   : rc / sigma_p, the fraction of risk each name carries (sums to 1)
      euler_gap : |sum(rc) - sigma_p| / sigma_p, zero up to rounding
    """
    x = np.asarray(x, dtype=float)
    Vx = _variance_gradient(x, cov)
    sigma_p = float(np.sqrt(max(x @ Vx, 0.0)))
    mrc = Vx / sigma_p
    rc = x * mrc
    return dict(
        sigma_p=sigma_p,
        mrc=mrc,
        rc=rc,
        share=rc / sigma_p,
        euler_gap=abs(rc.sum() - sigma_p) / sigma_p,
    )


def estimate_covs(weights_history, rets, mkt, rf, T: int = 60) -> dict:
    """Re-estimate the single-factor covariance used at each rebalance date.

    `rolling_backtest` does not store its covariance estimates, so they are
    rebuilt here from the same window and universe (the non-NaN columns of
    each weight row). Each stock's regression is independent of the others,
    so column order does not matter and the estimates are identical.
    """
    months = rets.index
    covs = {}
    for t in weights_history.index:
        universe = weights_history.loc[t].dropna().index
        i = months.get_loc(t)
        window = months[i - T + 1 : i + 1]
        excess = rets.loc[window, universe].values - rf.loc[window].values[:, None]
        covs[t] = estimate_single_factor(excess, mkt.loc[window].values, permno=universe.values)
    return covs


def risk_concentration(weights_history: pd.DataFrame, covs: dict) -> pd.DataFrame:
    """How concentrated is each month's ex-ante risk?

    For each rebalance date, with p_i = RC_i / sigma_p the share of risk name i
    carries:

      n_held          names with positive weight
      eff_n_weight    1 / sum x_i^2, the effective number of positions
      eff_n_risk      1 / sum p_i^2, the effective number of risk bets
      max_rc_share    largest single p_i
      top10_rc_share  share of risk carried by the 10 largest contributors
      rc_dispersion   max |n_held * p_i - 1| over held names (0 = risk parity)
      systematic_share  fraction of variance from the market factor,
                        sigmaM2 (beta'x)^2 / x'Vx (FactorCov only)
      pred_vol        ex-ante monthly volatility
    """
    rows = {}
    for t, cov in covs.items():
        x = weights_history.loc[t].reindex(cov.permno).fillna(0.0).values
        rc = risk_contributions(x, cov)
        p = rc["share"]
        held = x > 0
        row = dict(
            n_held=int(held.sum()),
            eff_n_weight=1.0 / float(np.sum(x**2)),
            eff_n_risk=1.0 / float(np.sum(p**2)),
            max_rc_share=float(p.max()),
            top10_rc_share=float(np.sort(p)[::-1][:10].sum()),
            rc_dispersion=float(np.abs(held.sum() * p[held] - 1.0).max()),
            pred_vol=rc["sigma_p"],
        )
        if isinstance(cov, FactorCov):
            row["systematic_share"] = cov.sigmaM2 * float(cov.beta @ x) ** 2 / rc["sigma_p"] ** 2
        rows[t] = row
    return pd.DataFrame(rows).T.rename_axis("rebalance")


# --------------------------------------------------------------------------
# Consistency across strategies
# --------------------------------------------------------------------------

def check_alignment(returns: pd.DataFrame, weights: dict, asset_returns: pd.DataFrame | None = None, strict: bool = True) -> list:
    """Verify every strategy is evaluated on the same months and universe.

    Checks
    ------
    1. every strategy has a return in every month, with no gaps
    2. every strategy rebalances on the same dates
    3. on each rebalance date every strategy draws from the same universe
    4. weights are fully invested and long only
    5. each return month is the month after a rebalance date (needs the panel index)

    Returns the list of problems found; raises AssertionError if `strict` and
    the list is non-empty.
    """
    problems = []

    missing = returns.isna().sum()
    for name, k in missing[missing > 0].items():
        problems.append(f"{name}: {k} months with no return")

    names = list(weights)
    ref = names[0]
    ref_dates = weights[ref].index
    for name in names[1:]:
        if not weights[name].index.equals(ref_dates):
            problems.append(f"{name}: rebalance dates differ from {ref}")

    for t in ref_dates:
        ref_universe = set(weights[ref].loc[t].dropna().index)
        for name in names[1:]:
            if t in weights[name].index and set(weights[name].loc[t].dropna().index) != ref_universe:
                problems.append(f"{name}: universe at {t} differs from {ref}")

    for name, w in weights.items():
        sums = w.sum(axis=1)
        if not np.allclose(sums, 1.0):
            problems.append(f"{name}: weights do not sum to 1 (range {sums.min():.6f}..{sums.max():.6f})")
        if (w.fillna(0.0) < -1e-12).any().any():
            problems.append(f"{name}: negative weights")

    if asset_returns is not None:
        months = asset_returns.index
        expected = pd.Index([months[months.get_loc(t) + 1] for t in ref_dates])
        if not returns.index.equals(expected):
            problems.append("return months are not the months following the rebalance dates")

    if strict and problems:
        raise AssertionError("alignment check failed:\n  " + "\n  ".join(problems))
    return problems
