"""Risk parity (equal risk contribution) portfolios.

Risk parity asks every name to carry the same total risk contribution,
RC_i = x_i (Vx)_i / sigma_p = sigma_p / n, with RC_i as defined by the Euler
decomposition in `backtesting_analysis.risk_contributions`. For long-only
portfolios this is the solution, rescaled to sum to one, of the strictly
convex problem
(Spinu 2013; Maillard, Roncalli and Teiletche 2010)

    min_y  1/2 y'Vy - sum_i log y_i,     y > 0

whose first-order condition y_i (Vy)_i = 1 says precisely that every name
contributes the same amount of variance.

Three solvers are provided:

  risk_parity_closed_form  single-factor V, one-dimensional root search (production)
  risk_parity_weights      any dense V, Newton's method on the problem above
  risk_parity_cvxpy        the same problem in cvxpy, for cross-validation only

`risk_parity` wraps the closed form in the constructor signature that
`backtest.rolling_backtest` expects.
"""

from __future__ import annotations

import numpy as np

from backtesting_analysis import risk_contributions
from estimation import FactorCov


def rc_dispersion(x, cov) -> float:
    """max_i |n * share_i - 1| over the held names: zero for exact risk parity."""
    share = risk_contributions(x, cov)["share"]
    held = np.asarray(x) > 0
    return float(np.abs(held.sum() * share[held] - 1.0).max())


def risk_parity_closed_form(cov: FactorCov, tol: float = 1e-14, maxiter: int = 300) -> np.ndarray:
    """Risk parity weights under a single-factor covariance.

    With V = sigmaM2 * beta beta' + Diag(omega2) and b = beta'y,

        y_i (Vy)_i = 1   <=>   omega2_i y_i^2 + (sigmaM2 beta_i b) y_i - 1 = 0,

    so for a given b each y_i is the positive root of its own quadratic. The
    whole solution is therefore pinned down by the scalar b, which must
    reproduce itself through b = beta'y(b). This is the same structure as the
    beta_LO threshold of the minimum-risk closed form.

    g(b) = beta'y(b) - b runs from +inf to -inf and has exactly one root,
    because every root yields a stationary point of the strictly convex
    problem in the module docstring. Bisection is therefore sufficient.
    """
    beta, omega2, sigmaM2 = cov.beta, cov.omega2, cov.sigmaM2

    def y_of(b):
        # Positive root of omega2 y^2 + a y - 1 = 0, written in whichever of the
        # two algebraically equivalent forms avoids cancellation for this sign of a.
        a = sigmaM2 * beta * b
        root = np.sqrt(a * a + 4.0 * omega2)
        return np.where(a >= 0, 2.0 / (a + root), (root - a) / (2.0 * omega2))

    def g(b):
        return float(beta @ y_of(b)) - b

    # Bracket the root starting from b = 0, in the direction g points.
    g0 = g(0.0)
    if g0 == 0.0:
        y = y_of(0.0)
        return y / y.sum()
    step = max(abs(g0), 1.0)
    low, high = (0.0, step) if g0 > 0 else (-step, 0.0)
    for _ in range(200):
        if g0 > 0 and g(high) < 0:
            break
        if g0 < 0 and g(low) > 0:
            break
        if g0 > 0:
            low, high = high, 2.0 * high
        else:
            low, high = 2.0 * low, low
    else:
        raise RuntimeError("could not bracket the risk parity root")

    # g is positive at `low` and negative at `high`.
    for _ in range(maxiter):
        mid = 0.5 * (low + high)
        if g(mid) > 0:
            low = mid
        else:
            high = mid
        if high - low < tol * max(1.0, abs(high)):
            break

    y = y_of(0.5 * (low + high))
    return y / y.sum()


def risk_parity_weights(Sigma, tol: float = 1e-9, maxiter: int = 100) -> np.ndarray:
    """Risk parity weights for an arbitrary dense covariance matrix.

    Newton's method on f(y) = 1/2 y'Sigma y - sum log y, with a backtracking
    line search that keeps y strictly positive. f is strictly convex, so the
    iteration converges to the unique solution, quadratically near the end.

    Starts from inverse-volatility weights scaled so that y'Sigma y = n, which
    the solution satisfies (sum_i y_i (Sigma y)_i = n).

    Stops when every name's variance contribution y_i (Sigma y)_i is within
    `tol` of 1, then rescales to sum to one.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be square, got shape {Sigma.shape}")

    y = 1.0 / np.sqrt(np.diag(Sigma))
    y *= np.sqrt(n / (y @ Sigma @ y))

    def f(v):
        return 0.5 * v @ Sigma @ v - np.log(v).sum()

    for _ in range(maxiter):
        Sy = Sigma @ y
        if np.abs(y * Sy - 1.0).max() < tol:
            break
        grad = Sy - 1.0 / y
        hess = Sigma + np.diag(1.0 / y**2)
        step = np.linalg.solve(hess, -grad)

        # Largest step that keeps y > 0.
        neg = step < 0
        t = min(1.0, 0.99 * float(np.min(-y[neg] / step[neg]))) if neg.any() else 1.0

        # Armijo backtracking while far from the solution. Close to it the
        # predicted decrease (the Newton decrement) falls below what f, of
        # order n, can resolve in floating point, so the line search would
        # stall; there pure Newton steps converge quadratically on their own.
        decrement = -float(grad @ step)
        if decrement > 1e-8:
            f0 = f(y)
            while f(y + t * step) > f0 - 1e-4 * t * decrement and t > 1e-12:
                t *= 0.5
        y = y + t * step
    else:
        raise RuntimeError("Newton's method did not converge")

    return y / y.sum()


def risk_parity_cvxpy(cov) -> np.ndarray:
    """The same convex problem solved with cvxpy. Cross-validation only.

    Accepts a FactorCov (objective in factor form) or a dense matrix.
    """
    import cvxpy as cp

    if isinstance(cov, FactorCov):
        n = cov.n
        y = cp.Variable(n, pos=True)
        quad = cov.sigmaM2 * cp.square(cov.beta @ y) + cp.sum_squares(
            cp.multiply(np.sqrt(cov.omega2), y)
        )
    else:
        Sigma = np.asarray(cov, dtype=float)
        n = Sigma.shape[0]
        y = cp.Variable(n, pos=True)
        quad = cp.quad_form(y, cp.psd_wrap(Sigma))

    problem = cp.Problem(cp.Minimize(0.5 * quad - cp.sum(cp.log(y))))
    # GUROBI is skipped on purpose: reviewers may not have a license.
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            problem.solve(solver=solver)
            if problem.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue
    else:
        raise RuntimeError(f"no solver succeeded (status={problem.status})")

    w = np.maximum(np.asarray(y.value, dtype=float), 0.0)
    return w / w.sum()


def risk_parity(cov: FactorCov, **_) -> dict:
    """Constructor for `backtest.rolling_backtest`.

    Returns the keys the backtest records (x, sigma2, n_held). Risk parity
    holds every name, so n_held is always n. Month-by-month risk-contribution
    diagnostics are computed afterwards by `backtesting_analysis.risk_concentration`.
    """
    x = risk_parity_closed_form(cov)
    return dict(x=x, sigma2=cov.quad(x), n_held=int((x > 0).sum()))
