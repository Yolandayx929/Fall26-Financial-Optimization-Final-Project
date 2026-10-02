"""Maximum diversification on the shared single-factor covariance model.

`max_diversification` solves the QP in Homework 4, Exercise 2(b).
`max_diversification_closed_form` uses the threshold in equation (4) for
monthly backtests. Both return fully invested, long-only portfolios.
"""

from __future__ import annotations

import numpy as np

from estimation import FactorCov


def _validate(cov: FactorCov) -> None:
    """Check the variances and model-implied asset volatilities."""
    if not isinstance(cov, FactorCov) or cov.n < 1:
        raise ValueError("expected a nonempty FactorCov")
    for name in ("beta", "omega2", "sigma"):
        values = np.asarray(getattr(cov, name))
        if values.shape != (cov.n,) or not np.isfinite(values).all():
            raise ValueError(f"invalid {name}")
    if not np.isfinite(cov.sigmaM2) or cov.sigmaM2 < 0 or (cov.omega2 <= 0).any():
        raise ValueError("market variance must be nonnegative; residual variances positive")
    expected = np.sqrt(cov.sigmaM2 * cov.beta**2 + cov.omega2)
    if not np.allclose(cov.sigma, expected, rtol=1e-10, atol=0):
        raise ValueError("sigma must match the covariance diagonal")


def _result(x: np.ndarray, cov: FactorCov, **extra) -> dict:
    return dict(x=x, sigma2=cov.quad(x), n_held=int((x > 0).sum()), **extra)


def max_diversification(cov: FactorCov, weight_tol: float = 1e-8, **_) -> dict:
    """QP reference: min y'Vy subject to sigma'y = 1, y >= 0.

    Use z_i = sigma_i*y_i for numerical scaling, then recover y and
    normalize x = y / sum(y). The objective stays in factor form.
    Small residual weights are removed, as in `portfolios.min_risk`.
    """
    import cvxpy as cp

    _validate(cov)
    z = cp.Variable(cov.n, nonneg=True)
    risk = cov.sigmaM2 * cp.square((cov.beta / cov.sigma) @ z)
    risk += cp.sum_squares(cp.multiply(np.sqrt(cov.omega2) / cov.sigma, z))
    problem = cp.Problem(cp.Minimize(risk), [cp.sum(z) == 1])
    _solve(problem)

    y = np.maximum(np.asarray(z.value, dtype=float), 0.0) / cov.sigma
    x = y / y.sum()
    x[x < weight_tol] = 0.0
    return _result(x / x.sum(), cov)


def _solve(problem) -> None:
    """Use an open solver, with tolerances for the numerical cross-check."""
    for solver, options in (
        ("CLARABEL", dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)),
        ("SCS", dict(eps=1e-8, max_iters=100000)),
    ):
        try:
            problem.solve(solver=solver, **options)
            if problem.status in ("optimal", "optimal_inaccurate"):
                return
        except Exception:
            continue
    raise RuntimeError(f"no solver succeeded (status={problem.status})")


def max_diversification_closed_form(
    cov: FactorCov, tol: float = 1e-13, maxiter: int = 300, **_
) -> dict:
    """HW4 equation (4): x_i proportional to sigma_i/omega2_i*(1-rho_i/h)^+.

    With rho_i = sigma_M*beta_i/sigma_i and d_i = omega2_i/sigma_i^2,
    the threshold h solves sum rho_i*(h-rho_i)^+/d_i = 1.
    Nonpositive betas use the QP; zero market variance gives inverse-vol.
    """
    _validate(cov)
    if cov.sigmaM2 == 0:
        u = 1.0 / cov.sigma
        return _result(u / u.sum(), cov)
    if (cov.beta <= 0).any():
        return max_diversification(cov)

    rho = np.sqrt(cov.sigmaM2) * cov.beta / cov.sigma
    d = cov.omega2 / cov.sigma**2

    def gap(threshold):
        return float(np.sum(rho * np.maximum(threshold - rho, 0.0) / d) - 1.0)

    low, high = float(rho.min()), 1.0
    for _ in range(200):
        if gap(high) > 0:
            break
        high *= 2.0
    else:
        raise RuntimeError("could not bracket rho_LO")

    for _ in range(maxiter):
        mid = 0.5 * (low + high)
        if gap(mid) > 0:
            high = mid
        else:
            low = mid
        if high - low <= tol * max(1.0, abs(mid)):
            break
    else:
        raise RuntimeError("MaxDiv bisection did not converge")

    threshold = 0.5 * (low + high)
    u = cov.sigma / cov.omega2 * np.maximum(1.0 - rho / threshold, 0.0)
    return _result(u / u.sum(), cov, rho_LO=threshold)


def diversification_ratio(x: np.ndarray, cov: FactorCov) -> float:
    """Weighted asset volatility divided by portfolio volatility."""
    x = np.asarray(x, dtype=float)
    if x.shape != (cov.n,) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("weights must be finite, aligned and nonnegative")
    if not np.isclose(x.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("weights must sum to one")
    return float(cov.sigma @ x / cov.exante_vol(x))


def optimality_residuals(x: np.ndarray, cov: FactorCov) -> dict:
    """Relative KKT residuals for the QP, used in numerical validation."""
    y = np.asarray(x) / float(cov.sigma @ x)
    vy = cov.sigmaM2 * cov.beta * (cov.beta @ y) + cov.omega2 * y
    q = cov.quad(y)
    slack = vy - q * cov.sigma
    scale = max(np.max(np.abs(vy)), np.max(np.abs(q * cov.sigma)), np.finfo(float).tiny)
    return dict(
        equality=abs(float(cov.sigma @ y) - 1.0),
        dual=max(0.0, float(-slack.min())) / scale,
        complementarity=float(np.max(np.abs(y * slack))) / q,
    )
