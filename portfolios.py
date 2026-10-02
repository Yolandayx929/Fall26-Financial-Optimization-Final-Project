"""Portfolio construction on a single-factor covariance estimate.

`min_risk` is the cvxpy model from Homework 4, Exercise 2(a).
`min_risk_closed_form` implements the analytic solution, equation (3) of the
same exercise, and exists to verify the optimizer numerically.

The equally weighted and value weighted benchmarks are built elsewhere; this module
covers the minimum risk portfolio only.

All portfolios here are fully invested and long only.
"""

from __future__ import annotations

import numpy as np

from estimation import FactorCov


def min_risk(cov: FactorCov, cap: float | None = None, weight_tol: float = 1e-5, **_) -> dict:
    """Minimum-variance portfolio.

        min  x'Vx
        s.t. 1'x = 1,  x >= 0  (and optionally x <= cap)

    The objective is written in factor form rather than as a quadratic form in
    a dense V. This is both faster and numerically safer, since the factor
    form is positive semidefinite by construction.

    Interior-point solvers return dust (order 1e-6) instead of exact zeros on
    the names the solution excludes, which would badly overstate the number of
    holdings. Weights below `weight_tol` are therefore set to zero and the
    vector renormalised; `min_risk_closed_form` is the authority on which
    names are genuinely held.

    Returns a dict with the weights, the attained variance, and the number of
    non-zero holdings.
    """
    import cvxpy as cp

    x = cp.Variable(cov.n, nonneg=True)
    constraints = [cp.sum(x) == 1]
    if cap is not None:
        constraints.append(x <= cap)

    # x'Vx = sigmaM2 * (beta'x)^2 + sum_i omega2_i * x_i^2
    risk = cov.sigmaM2 * cp.square(cov.beta @ x) + cp.sum_squares(
        cp.multiply(np.sqrt(cov.omega2), x)
    )

    problem = cp.Problem(cp.Minimize(risk), constraints)
    _solve(problem)

    weights = np.maximum(np.asarray(x.value, dtype=float), 0.0)
    weights[weights < weight_tol] = 0.0
    weights /= weights.sum()
    return dict(x=weights, sigma2=cov.quad(weights), n_held=int((weights > 0).sum()))


def _solve(problem) -> None:
    """Solve with whichever open solver is available.

    GUROBI is deliberately not used: peer reviewers have to be able to run the
    demo, and they may not have a license.
    """
    import cvxpy as cp

    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS, None):
        try:
            problem.solve(solver=solver) if solver else problem.solve()
            if problem.status in ("optimal", "optimal_inaccurate"):
                return
        except Exception:
            continue
    raise RuntimeError(f"no solver succeeded (status={problem.status})")


def min_risk_closed_form(cov: FactorCov, tol: float = 1e-14, maxiter: int = 300, **_) -> dict:
    """Analytic minimum-variance solution, Homework 4 equation (3).

        x_i = (sigma2_LMV / omega2_i) * (1 - beta_i / beta_LO)^+

    Only stocks with beta_i below the threshold beta_LO are held, so the whole
    solution is pinned down by that single scalar. For a candidate threshold
    we can form the implied weights and normalise them; the threshold is
    correct when it reproduces itself through

        beta_LO = x'Vx / (sigmaM2 * beta'x)

    The problem is a strictly convex QP, so its solution -- and therefore the
    root below -- is unique, and a bisection is enough to find it.
    """
    beta, omega2, sigmaM2 = cov.beta, cov.omega2, cov.sigmaM2
    inv_omega2 = 1.0 / omega2

    def weights(threshold):
        u = inv_omega2 * np.maximum(1.0 - beta / threshold, 0.0)
        total = u.sum()
        return None if total <= 0 else u / total

    def gap(threshold):
        """threshold minus the threshold it implies; zero at the solution."""
        x = weights(threshold)
        if x is None:
            return -np.inf
        portfolio_beta = float(beta @ x)
        if portfolio_beta <= 0:
            return -np.inf
        variance = sigmaM2 * portfolio_beta**2 + float(omega2 @ x**2)
        return threshold - variance / (sigmaM2 * portfolio_beta)

    # Just above min(beta) only the lowest-beta stock is held and the implied
    # threshold exceeds the candidate, so gap < 0. Expand upwards until the
    # sign flips, then bisect.
    low = max(float(beta.min()), 0.0)
    low += 1e-12 * max(1.0, abs(low))
    high = max(float(beta.max()), 2.0 * low)
    for _ in range(200):
        if gap(high) > 0:
            break
        high *= 2.0
    else:
        raise RuntimeError("could not bracket beta_LO")

    for _ in range(maxiter):
        mid = 0.5 * (low + high)
        if gap(mid) > 0:
            high = mid
        else:
            low = mid
        if high - low < tol * max(1.0, high):
            break

    threshold = 0.5 * (low + high)
    x = weights(threshold)
    return dict(
        x=x, sigma2=cov.quad(x), n_held=int((x > 0).sum()), beta_LO=threshold
    )
