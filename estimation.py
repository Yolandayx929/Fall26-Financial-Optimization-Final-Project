"""Single-factor covariance estimation, following the approach in Homework 3.

Model (market model / single-index model, estimated on excess returns):

    r_i - r_f = alpha_i + beta_i * (r_M - r_f) + eps_i

    V = sigmaM2 * beta beta' + Diag(omega2)

The dense matrix V is never built during the backtest. Every quantity the
portfolio optimizers need is carried by the triple (beta, omega2, sigmaM2),
which keeps the factor structure available to downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FactorCov:
    """Single-factor covariance estimate for one rebalance date.

    Attributes
    ----------
    permno : identifiers of the n stocks, aligned with every other array.
    beta : shrunk betas (what the covariance model uses).
    omega2 : residual (idiosyncratic) variances.
    sigmaM2 : variance of the market excess return.
    sigma : model-implied total volatility, sqrt(sigmaM2 * beta^2 + omega2).
        Maximum diversification must use this, not a sample standard
        deviation, or sigma and V would be mutually inconsistent.
    beta_hat : raw OLS betas before shrinkage, kept for diagnostics.
    alpha : OLS intercepts, kept for diagnostics.
    """

    permno: np.ndarray
    beta: np.ndarray
    omega2: np.ndarray
    sigmaM2: float
    sigma: np.ndarray
    beta_hat: np.ndarray
    alpha: np.ndarray

    @property
    def n(self) -> int:
        return len(self.permno)

    def quad(self, x) -> float:
        """Portfolio variance x'Vx, evaluated in factor form."""
        x = np.asarray(x, dtype=float)
        return self.sigmaM2 * float(self.beta @ x) ** 2 + float(self.omega2 @ x**2)

    def exante_vol(self, x) -> float:
        """Predicted portfolio volatility sqrt(x'Vx)."""
        return float(np.sqrt(max(self.quad(x), 0.0)))

    def exante_beta(self, x) -> float:
        """Predicted portfolio beta against the market proxy."""
        return float(self.beta @ np.asarray(x, dtype=float))

    def to_dense(self) -> np.ndarray:
        """Materialise V. Only needed for tests and cross-checks."""
        return self.sigmaM2 * np.outer(self.beta, self.beta) + np.diag(self.omega2)


def estimate_single_factor(R, rM, permno=None) -> FactorCov:
    """Estimate the single-factor covariance model on one estimation window.

    Parameters
    ----------
    R : array of shape (T, n)
        Stock excess returns. Must be complete -- selecting a universe with
        no missing observations is the caller's responsibility.
    rM : array of shape (T,)
        Market excess return over the same window.
    permno : array of shape (n,), optional
        Stock identifiers carried through to the result.

    Notes
    -----
    All n regressions are run at once as a single least-squares solve, as in
    Homework 3. Betas are then shrunk with the fixed-weight Blume rule
    beta~ = (2/3) * beta_hat + 1/3: estimated betas are noisy and true betas
    tend to revert towards one, so the raw cross-sectional spread of beta_hat
    is too wide.
    """
    R = np.asarray(R, dtype=float)
    rM = np.asarray(rM, dtype=float).ravel()
    T, n = R.shape

    if len(rM) != T:
        raise ValueError(f"rM has length {len(rM)}, expected T={T}")
    if not np.isfinite(R).all():
        raise ValueError("R contains NaN or inf; clean the universe upstream")
    if T <= 2:
        raise ValueError(f"estimation window too short: T={T}")

    # One least-squares solve for all n stocks: B is 2 x n, rows [alpha; beta].
    X = np.column_stack([np.ones(T), rM])
    B = np.linalg.solve(X.T @ X, X.T @ R)
    alpha, beta_hat = B[0], B[1]

    # Residual variance. T - 2 degrees of freedom because alpha and beta were
    # both estimated. Floored to keep V strictly positive definite.
    residuals = R - X @ B
    omega2 = (residuals**2).sum(axis=0) / (T - 2)
    omega2 = np.maximum(omega2, 1e-12)

    sigmaM2 = float(rM.var(ddof=1))

    beta = (2.0 / 3.0) * beta_hat + 1.0 / 3.0
    sigma = np.sqrt(sigmaM2 * beta**2 + omega2)

    return FactorCov(
        permno=np.arange(n) if permno is None else np.asarray(permno),
        beta=beta,
        omega2=omega2,
        sigmaM2=sigmaM2,
        sigma=sigma,
        beta_hat=beta_hat,
        alpha=alpha,
    )
