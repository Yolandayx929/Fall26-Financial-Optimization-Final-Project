"""Equally weighted and value weighted benchmark portfolios.

Both are constructors for `backtest.rolling_backtest`: they receive the same universe and
covariance estimate as every other strategy, and return the keys the backtest records
(x, sigma2, n_held). Neither uses the covariance to set weights; it is only used to report
the predicted variance, so the benchmarks sit in the same tables as the optimised portfolios.
"""

from __future__ import annotations

import numpy as np

from estimation import FactorCov


def equal_weight(cov: FactorCov, **_) -> dict:
    """1/n in every stock of the universe."""
    x = np.full(cov.n, 1.0 / cov.n)
    return dict(x=x, sigma2=cov.quad(x), n_held=cov.n)


def value_weight(cov: FactorCov, caps=None, **_) -> dict:
    """Weights proportional to market cap at the rebalance date.

    `rolling_backtest` passes caps aligned with cov.permno; select_universe guarantees
    they are positive.
    """
    if caps is None:
        raise ValueError("value_weight needs the market caps at the rebalance date")
    caps = np.asarray(caps, dtype=float)
    if len(caps) != cov.n or not (np.isfinite(caps) & (caps > 0)).all():
        raise ValueError("caps must be positive and aligned with cov.permno")
    x = caps / caps.sum()
    return dict(x=x, sigma2=cov.quad(x), n_held=cov.n)
