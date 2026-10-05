"""Anonymous Topic 3 demo: comparison of diversification approaches.

Reading order
-------------
1. Shared data loading, cleaning and investable-universe selection.
2. Shared single-factor covariance model and parameter estimation, plus the
   two alternative estimators from project.pdf: constant correlation and the
   shrunk sample covariance.
3. Independent portfolio rules: EW/VW, minimum risk, MaxDiv, risk parity
   (3E: the same rules on the shrunk sample covariance).
4. Shared rolling out-of-sample backtest.
5. Shared performance metrics, risk diagnostics and alignment checks.
6. Report tables and figures shared by every covariance estimator.

Each portfolio rule consumes the same covariance estimate and returns x,
sigma2 and n_held. The rules can be reviewed independently; the backtest
evaluates them on identical dates and universes. Risk-parity diagnostics use
the shared risk_contributions function in section 5, resolved when called
after import.

Run project_demo.ipynb for the complete experiment. Optional optimization
references import cvxpy only when used; no commercial solver is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# 1. Shared data and universe selection
# ======================================================================
#
# Topic 3 data layer: download and clean CRSP, data-quality checks, universe summary.
#
# Produces data/crsp_msf_v2.parquet, the file every notebook in analysis/ reads.
# Universe selection itself is select_universe (the team standard); this module
# only summarises the universe it picks.
#
# Documented rules
# ----------------
# Data source   crsp.msf_v2 (CIZ format, annual update), 1990-01 to 2025-12. In CIZ, mthret
#               already includes the delisting return in a stock's final month.
# Stocks        US common stocks, filtered in the SQL query: sharetype='NS',
#               securitytype='EQTY', securitysubtype='COM', usincflg='Y',
#               issuertype in ('ACOR','CORP'). The exchange filter (primaryexch in N, A, Q)
#               is applied by the notebooks after loading.
# Market cap    mthcap, in $ thousands (column mcap holds the same value in $ millions).






# ---------------------------------------------------------------------------
# Download and cache
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
# Data-quality checks and universe summary
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

    rets, caps : month x permno panels as used by rolling_backtest
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




# ======================================================================
# 2. Shared single-factor covariance model
# ======================================================================
# Single-factor covariance estimation, following the approach in Homework 3.
#
# Model (market model / single-index model, estimated on excess returns):
#
#     r_i - r_f = alpha_i + beta_i * (r_M - r_f) + eps_i
#
#     V = sigmaM2 * beta beta' + Diag(omega2)
#
# The dense matrix V is never built during the backtest. Every quantity the
# portfolio optimizers need is carried by the triple (beta, omega2, sigmaM2),
# which keeps the factor structure available to downstream code.





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


# ----------------------------------------------------------------------
# Alternative estimators (last page of project.pdf)
# ----------------------------------------------------------------------
#
#     constant correlation   V = rho * s s' + (1 - rho) * Diag(s)^2
#     shrunk sample          V = Vhat + lambda * (C - Vhat),   Vhat = R'R
#
# The constant-correlation matrix is again "rank one plus diagonal", so it is
# returned as a FactorCov with sigmaM2 = 1 and beta = sqrt(rho) * s, and every
# closed-form optimizer applies unchanged. Its `beta` field is then a loading
# on the common correlation component, not a market beta; the backtest takes
# market betas from the single-factor regression whatever the risk model.


def _check_window(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] <= 2:
        raise ValueError(f"expected a (T, n) window with T > 2, got shape {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError("R contains NaN or inf; clean the universe upstream")
    return R


def estimate_constant_correlation(R, permno=None) -> FactorCov:
    """Constant-correlation covariance on one estimation window.

    rho is the average of all pairwise sample correlations. Volatilities are
    sample standard deviations whose logs are shrunk 1/3 towards their
    cross-sectional mean. rho can be recovered as (beta / sigma)^2.
    """
    R = _check_window(R)
    T, n = R.shape
    s = R.std(axis=0, ddof=1)

    # The entries of the correlation matrix sum to ||Z 1||^2 / (T - 1), with Z
    # the standardised returns; removing the n diagonal ones leaves the
    # pairwise correlations, without forming the n x n matrix.
    Z = (R - R.mean(axis=0)) / s
    rho = (float(np.sum(Z.sum(axis=1) ** 2)) / (T - 1) - n) / (n * (n - 1))
    if not 0.0 < rho < 1.0:
        raise ValueError(f"average correlation {rho:.3f} is outside (0, 1)")

    log_s = np.log(s)
    s = np.exp((2.0 / 3.0) * log_s + (1.0 / 3.0) * log_s.mean())

    unused = np.full(n, np.nan)
    return FactorCov(
        permno=np.arange(n) if permno is None else np.asarray(permno),
        beta=np.sqrt(rho) * s,
        omega2=(1.0 - rho) * s**2,
        sigmaM2=1.0,
        sigma=s,
        beta_hat=unused,
        alpha=unused,
    )


@dataclass
class ShrunkSampleCov:
    """Sample covariance shrunk towards a constant target (CdST 2006 appendix).

    With R the (T, n) window of excess returns, Vhat = R'R and the target
    C = (dbar - cbar) I + cbar 11', the estimate (Vhat + lam (C - Vhat)) / T is
    stored as

        V = a R'R + b I + c 11',   a = (1-lam)/T, b = lam (dbar-cbar)/T, c = lam cbar/T

    so x'Vx and Vx cost O(nT) and the optimizers never need the dense matrix.
    Dividing by T only fixes the scale of predicted volatility; no portfolio
    weight depends on it.
    """

    permno: np.ndarray
    R: np.ndarray
    lam: float
    a: float
    b: float
    c: float
    sigma: np.ndarray

    @property
    def n(self) -> int:
        return len(self.permno)

    def matvec(self, x) -> np.ndarray:
        """Vx."""
        x = np.asarray(x, dtype=float)
        return self.a * (self.R.T @ (self.R @ x)) + self.b * x + self.c * x.sum()

    def quad(self, x) -> float:
        """Portfolio variance x'Vx."""
        x = np.asarray(x, dtype=float)
        Rx = self.R @ x
        return self.a * float(Rx @ Rx) + self.b * float(x @ x) + self.c * float(x.sum()) ** 2

    def exante_vol(self, x) -> float:
        return float(np.sqrt(max(self.quad(x), 0.0)))

    def to_dense(self) -> np.ndarray:
        return self.a * (self.R.T @ self.R) + self.b * np.eye(self.n) + self.c


def estimate_shrunk_sample(R, permno=None) -> ShrunkSampleCov:
    """Shrunk sample covariance on one estimation window, as in project.pdf.

    Sample means are ignored (Vhat = R'R), as the PDF and the CdST appendix do.
    The intensity is lam = max(0, min(1, lam_bar)) with

        lam_bar = [sum_ij sum_t r_it^2 r_jt^2 - (1/T) sum_ij (sum_t r_it r_jt)^2]
                  / trace((Vhat - C)^2)

    The first sum equals sum_t (sum_i r_it^2)^2 and the second is the squared
    Frobenius norm of Vhat, so neither needs a loop over pairs.
    """
    R = _check_window(R)
    T, n = R.shape
    Vhat = R.T @ R
    diag = np.diag(Vhat)
    dbar = float(diag.mean())
    cbar = float((Vhat.sum() - diag.sum()) / (n * (n - 1)))

    numerator = float(np.sum((R**2).sum(axis=1) ** 2)) - float(np.sum(Vhat**2)) / T
    gap = Vhat - cbar
    gap[np.diag_indices(n)] = diag - dbar
    lam = float(np.clip(numerator / float(np.sum(gap**2)), 0.0, 1.0))

    a, b, c = (1.0 - lam) / T, lam * (dbar - cbar) / T, lam * cbar / T
    if b <= 0:
        # Without a positive multiple of I the matrix has rank <= T + 1 < n.
        raise ValueError(f"shrunk covariance is singular (lam={lam:.3f})")
    return ShrunkSampleCov(
        permno=np.arange(n) if permno is None else np.asarray(permno),
        R=R,
        lam=lam,
        a=a,
        b=b,
        c=c,
        sigma=np.sqrt(a * diag + b + c),
    )


ESTIMATORS = ("single_factor", "const_corr", "shrunk_sample")


def estimate_covariance(estimator: str, R, rM, permno=None):
    """Dispatch to one of ESTIMATORS. rM is only used by the single-factor model."""
    if estimator == "single_factor":
        return estimate_single_factor(R, rM, permno=permno)
    if estimator == "const_corr":
        return estimate_constant_correlation(R, permno=permno)
    if estimator == "shrunk_sample":
        return estimate_shrunk_sample(R, permno=permno)
    raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")



# ======================================================================
# 3A. Parallel strategies: equal and value weighting
# ======================================================================
# Equally weighted and value weighted benchmark portfolios.
#
# Both are constructors for `rolling_backtest`: they receive the same universe and
# covariance estimate as every other strategy, and return the keys the backtest records
# (x, sigma2, n_held). Neither uses the covariance to set weights; it is only used to report
# the predicted variance, so the benchmarks sit in the same tables as the optimised portfolios.





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



# ======================================================================
# 3B. Parallel strategies: minimum risk
# ======================================================================
# Portfolio construction on a single-factor covariance estimate.
#
# `min_risk` is the cvxpy model from Homework 4, Exercise 2(a).
# `min_risk_closed_form` implements the analytic solution, equation (3) of the
# same exercise, and exists to verify the optimizer numerically.
#
# The equally weighted and value weighted benchmarks are built elsewhere; this module
# covers the minimum risk portfolio only.
#
# All portfolios here are fully invested and long only.





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
    _min_risk_solve(problem)

    weights = np.maximum(np.asarray(x.value, dtype=float), 0.0)
    weights[weights < weight_tol] = 0.0
    weights /= weights.sum()
    return dict(x=weights, sigma2=cov.quad(weights), n_held=int((weights > 0).sum()))


def _min_risk_solve(problem) -> None:
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



# ======================================================================
# 3C. Parallel strategies: maximum diversification
# ======================================================================
# Maximum diversification on the shared single-factor covariance model.
#
# `max_diversification` solves the QP in Homework 4, Exercise 2(b).
# `max_diversification_closed_form` uses the threshold in equation (4) for
# monthly backtests. Both return fully invested, long-only portfolios.





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
    Small residual weights are removed, as in `min_risk`.
    """
    import cvxpy as cp

    _validate(cov)
    z = cp.Variable(cov.n, nonneg=True)
    risk = cov.sigmaM2 * cp.square((cov.beta / cov.sigma) @ z)
    risk += cp.sum_squares(cp.multiply(np.sqrt(cov.omega2) / cov.sigma, z))
    problem = cp.Problem(cp.Minimize(risk), [cp.sum(z) == 1])
    _max_diversification_solve(problem)

    y = np.maximum(np.asarray(z.value, dtype=float), 0.0) / cov.sigma
    x = y / y.sum()
    x[x < weight_tol] = 0.0
    return _result(x / x.sum(), cov)


def _max_diversification_solve(problem) -> None:
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
    """Relative KKT residuals for the QP, used in numerical validation.

    Works for any covariance object accepted by `risk_contributions`.
    """
    y = np.asarray(x) / float(cov.sigma @ x)
    vy = _variance_gradient(y, cov)
    q = float(y @ vy)
    slack = vy - q * cov.sigma
    scale = max(np.max(np.abs(vy)), np.max(np.abs(q * cov.sigma)), np.finfo(float).tiny)
    return dict(
        equality=abs(float(cov.sigma @ y) - 1.0),
        dual=max(0.0, float(-slack.min())) / scale,
        complementarity=float(np.max(np.abs(y * slack))) / q,
    )



# ======================================================================
# 3D. Parallel strategies: risk parity
# ======================================================================
# Risk parity (equal risk contribution) portfolios.
#
# Risk parity asks every name to carry the same total risk contribution,
# RC_i = x_i (Vx)_i / sigma_p = sigma_p / n, with RC_i as defined by the Euler
# decomposition in `risk_contributions`. For long-only
# portfolios this is the solution, rescaled to sum to one, of the strictly
# convex problem
# (Spinu 2013; Maillard, Roncalli and Teiletche 2010)
#
#     min_y  1/2 y'Vy - sum_i log y_i,     y > 0
#
# whose first-order condition y_i (Vy)_i = 1 says precisely that every name
# contributes the same amount of variance.
#
# Three solvers are provided:
#
#   risk_parity_closed_form  single-factor V, one-dimensional root search (production)
#   risk_parity_weights      any dense V, Newton's method on the problem above
#   risk_parity_cvxpy        the same problem in cvxpy, for cross-validation only
#
# `risk_parity` wraps the closed form in the constructor signature that
# `rolling_backtest` expects.





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
    problem described in the risk-parity section. Bisection is therefore sufficient.
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
    """Constructor for `rolling_backtest`.

    Returns the keys the backtest records (x, sigma2, n_held). Risk parity
    holds every name, so n_held is always n. Month-by-month risk-contribution
    diagnostics are computed afterwards by `risk_concentration`.
    """
    x = risk_parity_closed_form(cov)
    return dict(x=x, sigma2=cov.quad(x), n_held=int((x > 0).sum()))


# ======================================================================
# 3E. The same rules on the shrunk sample covariance
# ======================================================================
# The constant-correlation estimate is a FactorCov, so it reuses the closed
# forms above. The shrunk sample covariance has no one-factor structure: the
# minimum-risk and MaxDiv QPs are solved with cvxpy in the low-rank form
# a ||R x||^2 + b ||x||^2 + c (1'x)^2, and risk parity uses Newton's method on
# the dense matrix (risk_parity_weights).


def _shrunk_quad(cov: ShrunkSampleCov, x, scale=None):
    """cvxpy expression for x'Vx, or for (x/scale)'V(x/scale)."""
    import cvxpy as cp

    R = cov.R if scale is None else cov.R / scale
    y = x if scale is None else cp.multiply(1.0 / scale, x)
    return cov.a * cp.sum_squares(R @ x) + cov.b * cp.sum_squares(y) + cov.c * cp.square(cp.sum(y))


def min_risk_shrunk(cov: ShrunkSampleCov, weight_tol: float = 1e-5, **_) -> dict:
    """min x'Vx s.t. 1'x = 1, x >= 0; dust below weight_tol removed as in `min_risk`."""
    import cvxpy as cp

    x = cp.Variable(cov.n, nonneg=True)
    problem = cp.Problem(cp.Minimize(_shrunk_quad(cov, x)), [cp.sum(x) == 1])
    _max_diversification_solve(problem)

    w = np.maximum(np.asarray(x.value, dtype=float), 0.0)
    w[w < weight_tol] = 0.0
    w /= w.sum()
    return dict(x=w, sigma2=cov.quad(w), n_held=int((w > 0).sum()))


def max_diversification_shrunk(cov: ShrunkSampleCov, weight_tol: float = 1e-8, **_) -> dict:
    """Same QP as `max_diversification`: min y'Vy s.t. sigma'y = 1, y >= 0,
    solved in z = sigma * y for scaling."""
    import cvxpy as cp

    z = cp.Variable(cov.n, nonneg=True)
    problem = cp.Problem(cp.Minimize(_shrunk_quad(cov, z, scale=cov.sigma)), [cp.sum(z) == 1])
    _max_diversification_solve(problem)

    y = np.maximum(np.asarray(z.value, dtype=float), 0.0) / cov.sigma
    x = y / y.sum()
    x[x < weight_tol] = 0.0
    x /= x.sum()
    return dict(x=x, sigma2=cov.quad(x), n_held=int((x > 0).sum()))


def risk_parity_shrunk(cov: ShrunkSampleCov, **_) -> dict:
    x = risk_parity_weights(cov.to_dense())
    return dict(x=x, sigma2=cov.quad(x), n_held=int((x > 0).sum()))


def strategies(estimator: str = "single_factor") -> dict:
    """The five constructors for one covariance estimator, with identical keys."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")
    if estimator == "shrunk_sample":
        optimized = dict(min_risk=min_risk_shrunk, risk_parity=risk_parity_shrunk,
                         max_div=max_diversification_shrunk)
    else:
        optimized = dict(min_risk=min_risk_closed_form, risk_parity=risk_parity,
                         max_div=max_diversification_closed_form)
    return dict(ew=equal_weight, vw=value_weight, **optimized)



# ======================================================================
# 4. Shared rolling out-of-sample backtest
# ======================================================================
# Rolling out-of-sample backtest.
#
# At every rebalance date t the procedure is:
#
#   1. select the universe using only information known at t,
#   2. estimate the covariance model on the previous T months,
#   3. build each portfolio,
#   4. score it on the return of month t+1.
#
# Expected inputs (monthly, in decimals, not percentage points):
#
#   rets : DataFrame, index = month-end dates ascending, columns = permno,
#          values = total return
#   caps : DataFrame, same shape as rets, values = market capitalisation
#   mkt  : Series indexed like rets, market excess return
#   rf   : Series indexed like rets, risk-free rate




# The closed form is ~3900x faster than the cvxpy model on n=500 (0.2ms vs 776ms),
# which turns a 372-month backtest from ~5 minutes into a fraction of a second.
# `min_risk` from project.py solves the same problem and is kept for
# cross-validation.
CONSTRUCTORS = {"min_risk": min_risk_closed_form}




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
    estimator: str = "single_factor",
):
    """Run the rolling backtest.

    `estimator` selects the risk model the constructors receive (see
    ESTIMATORS; use the matching `strategies(estimator)`). The single-factor
    regression is run in every case, because the recorded ex-ante beta is
    always the market beta from that regression.

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
        model = cov if estimator == "single_factor" else estimate_covariance(
            estimator, excess, None, permno=universe.values
        )

        # Out-of-sample month. If a holding has no return at t+1 its weight is
        # assumed to earn the risk-free rate. Renormalising over the survivors
        # instead would amount to knowing at the start of the month which
        # stocks were about to disappear.
        r_next = rets.loc[t_next, universe].fillna(float(rf.loc[t_next])).values

        for name, construct in constructors.items():
            # caps are passed through for constructors that need them; those that
            # do not simply absorb the keyword.
            result = construct(model, caps=caps.loc[t, universe].values)
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
                    exante_beta=cov.exante_beta(x),  # single-factor market beta
                    n_held=result["n_held"],
                    # only the single-factor closed form has a beta threshold
                    beta_LO=result.get("beta_LO", np.nan) if model is cov else np.nan,
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
        drawdown = cumulative / np.maximum(1.0, np.maximum.accumulate(cumulative)) - 1.0
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



# ======================================================================
# 5. Shared performance and risk diagnostics
# ======================================================================
# Backtesting analysis, applied identically to every strategy.
#
# Covers performance metrics, turnover, risk decomposition and concentration
# diagnostics, and the cross-strategy alignment checks.
#
# Conventions, chosen to match `summarize` exactly so the two can be
# cross-checked:
#
#   * returns are monthly decimals, indexed by the month they were earned (yyyymm)
#   * annualised return is geometric
#   * annualised volatility and the Sharpe ratio use *excess* returns
#   * maximum drawdown is taken on the compounded total-return path
#   * turnover is one-way, 0.5 * sum |w_t - w_{t-1}^drifted|, measured against
#     the previous weights *after* they drifted with realised returns, so that
#     passive drift is not counted as trading; the first rebalance has none
#   * a holding with no return in a month earns the risk-free rate
#
# Weights histories are DataFrames indexed by rebalance date t, with one column
# per permno and NaN for names outside that date's universe -- the format that
# `rolling_backtest` returns. A weight set at t earns the return of the
# next month in the return panel.





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
    """Vx, for a FactorCov, a ShrunkSampleCov or a dense matrix."""
    x = np.asarray(x, dtype=float)
    if isinstance(cov, FactorCov):
        return cov.sigmaM2 * cov.beta * float(cov.beta @ x) + cov.omega2 * x
    if isinstance(cov, ShrunkSampleCov):
        return cov.matvec(x)
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
    cov : FactorCov, ShrunkSampleCov or dense covariance matrix of shape (n, n).

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


def estimate_covs(weights_history, rets, mkt, rf, T: int = 60,
                  estimator: str = "single_factor") -> dict:
    """Re-estimate the covariance used at each rebalance date.

    `rolling_backtest` does not store its covariance estimates, so they are
    rebuilt here from the same window and universe (the non-NaN columns of
    each weight row). Every estimator treats stocks symmetrically, so column
    order does not matter and the estimates are identical.
    """
    months = rets.index
    covs = {}
    for t in weights_history.index:
        universe = weights_history.loc[t].dropna().index
        i = months.get_loc(t)
        window = months[i - T + 1 : i + 1]
        excess = rets.loc[window, universe].values - rf.loc[window].values[:, None]
        covs[t] = estimate_covariance(estimator, excess, mkt.loc[window].values, permno=universe.values)
    return covs


def risk_concentration(weights_history: pd.DataFrame, covs: dict,
                       factor_covs: dict | None = None) -> pd.DataFrame:
    """How concentrated is each month's ex-ante risk?

    Risk contributions use `covs`, the model the portfolios were built on. The
    market share of variance is measured with the single-factor model: `covs`
    itself by default, or `factor_covs` (same dates and universes) when `covs`
    comes from another estimator, so the column is comparable across models.

    For each rebalance date, with p_i = RC_i / sigma_p the share of risk name i
    carries:

      n_held          names with positive weight
      eff_n_weight    1 / sum x_i^2, the effective number of positions
      eff_n_risk      1 / sum p_i^2, the effective number of risk bets
      max_rc_share    largest single p_i
      top10_rc_share  share of risk carried by the 10 largest contributors
      rc_dispersion   max |n_held * p_i - 1| over held names (0 = risk parity)
      systematic_share  fraction of variance from the market factor,
                        sigmaM2 (beta'x)^2 / x'Vx under the single-factor model
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
        if factor_covs is None:
            if isinstance(cov, FactorCov):
                row["systematic_share"] = cov.sigmaM2 * float(cov.beta @ x) ** 2 / rc["sigma_p"] ** 2
        else:
            market = factor_covs[t]
            xm = weights_history.loc[t].reindex(market.permno).fillna(0.0).values
            row["systematic_share"] = market.sigmaM2 * float(market.beta @ xm) ** 2 / market.quad(xm)
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


# ======================================================================
# 6. Report tables and figures
# ======================================================================
# Used by the notebook for the single-factor results (sections 6-8) and again,
# unchanged, for each alternative covariance estimator. matplotlib is imported
# only when a figure is drawn.

COLORS = {"max_div": "#eb6834", "min_risk": "#2a78d6", "risk_parity": "#1baf7a",
          "ew": "#eda100", "vw": "#e87ba4"}
LABELS = {"max_div": "Maximum diversification", "min_risk": "Minimum risk",
          "risk_parity": "Risk parity", "ew": "Equally weighted", "vw": "Value weighted"}
ESTIMATOR_COLORS = {"single_factor": "#1f2d5c", "const_corr": "#8e44ad", "shrunk_sample": "#16a3b8"}
ESTIMATOR_LABELS = {"single_factor": "Single factor", "const_corr": "Constant correlation",
                    "shrunk_sample": "Shrunk sample"}
CONCENTRATION_COLUMNS = ["div_ratio", "n_held", "eff_n_weight", "eff_n_risk",
                         "top10_rc_share", "systematic_share"]


def _month_starts(index) -> pd.DatetimeIndex:
    """yyyymm integers to timestamps for plotting."""
    return pd.PeriodIndex([str(d) for d in index], freq="M").to_timestamp()


def plot_cumulative(returns: pd.DataFrame, title: str, colors=None, labels=None):
    """Growth of $1 on a log scale, one line per column."""
    import matplotlib.pyplot as plt

    colors, labels = colors or COLORS, labels or LABELS
    when = _month_starts(returns.index)
    wealth = 1.0 + cumulative_returns(returns)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k in returns.columns:
        ax.plot(when, wealth[k], color=colors.get(k),
                label=f"{labels.get(k, k)} (${wealth[k].iloc[-1]:.1f})")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_drawdowns(returns: pd.DataFrame, title: str = "Drawdown from running peak",
                   colors=None, labels=None):
    import matplotlib.pyplot as plt

    colors, labels = colors or COLORS, labels or LABELS
    when = _month_starts(returns.index)
    dd = drawdowns(returns)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k in returns.columns:
        ax.plot(when, dd[k], color=colors[k], label=labels[k])
    ax.set(ylabel="Drawdown", title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def concentration_diagnostics(weights: dict, covs: dict, factor_covs: dict | None = None) -> dict:
    """`risk_concentration` plus the diversification ratio, per strategy."""
    conc = {k: risk_concentration(w, covs, factor_covs) for k, w in weights.items()}
    for k, w in weights.items():
        conc[k]["div_ratio"] = pd.Series({
            t: diversification_ratio(w.loc[t].reindex(c.permno).values, c)
            for t, c in covs.items()
        })
    return conc


def diversification_checks(weights: dict, covs: dict, conc: dict) -> dict:
    """Formation-date checks, under the covariance the portfolios were built on.

    dr_excess          max over dates and strategies of DR - DR(MaxDiv); <= 0 up to rounding
    max_kkt            largest relative KKT residual of the MaxDiv QP
    max_rp_dispersion  largest deviation from equal risk contributions in risk parity
    """
    dr = pd.DataFrame({k: c.div_ratio for k, c in conc.items()})
    return dict(
        dr_excess=float(dr.sub(dr["max_div"], axis=0).to_numpy().max()),
        max_kkt=max(max(optimality_residuals(
            weights["max_div"].loc[t].reindex(c.permno).values, c).values())
            for t, c in covs.items()),
        max_rp_dispersion=max(rc_dispersion(
            weights["risk_parity"].loc[t].reindex(c.permno).values, c)
            for t, c in covs.items()),
    )


def concentration_table(conc: dict) -> pd.DataFrame:
    """Time-averaged concentration diagnostics, one row per strategy."""
    return pd.DataFrame({LABELS[k]: c[CONCENTRATION_COLUMNS].mean() for k, c in conc.items()}).T


def plot_diversification(conc: dict, title_suffix: str = ""):
    """Diversification ratio and market share of variance at each formation date."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, c in conc.items():
        when = _month_starts(c.index)
        ax[0].plot(when, c.div_ratio, color=COLORS[k], label=LABELS[k])
        ax[1].plot(when, c.systematic_share, color=COLORS[k], label=LABELS[k])
    ax[0].set(ylabel="Diversification ratio", title="Estimated diversification at formation" + title_suffix)
    ax[1].set(ylabel="Market share of variance", title="Factor concentration" + title_suffix)
    ax[0].legend(frameon=False); ax[1].legend(frameon=False)
    fig.tight_layout()
    return fig


def sector_weight(weights_history: pd.DataFrame, sic: pd.DataFrame,
                  lo: int = 4900, hi: int = 4999) -> pd.Series:
    """Weight in SIC codes lo..hi at each rebalance date (default: utilities, SIC 49)."""
    codes = sic.reindex(index=weights_history.index, columns=weights_history.columns).astype(float)
    return weights_history.where((codes >= lo) & (codes <= hi)).sum(axis=1)


def estimator_comparison(runs: dict, *, rf, mkt, sic,
                         optimized=("min_risk", "max_div", "risk_parity")) -> pd.DataFrame:
    """Each optimized strategy under each estimator, with EW and VW as references.

    runs maps estimator -> dict(returns=..., weights=..., panel=...). EW and VW do
    not use a covariance; their predicted/realized ratio uses the single-factor model.
    """
    def row(estimator, k):
        run = runs[estimator]
        perf = performance_metrics(run["returns"][k], rf=rf, mkt=mkt)
        ref = summarize(run["panel"]).loc[k]
        return {
            "CAGR (%)": 100 * perf["ann_return"],
            "Volatility (%)": 100 * perf["ann_vol"],
            "Sharpe": perf["sharpe"],
            "Max drawdown (%)": 100 * perf["max_drawdown"],
            "Realized beta": perf["beta"],
            "Turnover (%)": 100 * ref["ann_turnover"],
            "Pred / realized vol": ref["pred_over_realized"],
            "Avg holdings": ref["avg_n_held"],
            "Utilities weight (%)": 100 * sector_weight(run["weights"][k], sic).mean(),
        }

    rows = {(LABELS[k], ESTIMATOR_LABELS[e]): row(e, k) for k in optimized for e in runs}
    for k in ("ew", "vw"):
        rows[(LABELS[k], "Benchmark")] = row("single_factor", k)
    return pd.DataFrame(rows).T.rename_axis(["Strategy", "Risk model"])


def estimator_returns(runs: dict, strategy: str) -> pd.DataFrame:
    """One strategy's monthly returns under every estimator, plus EW and VW."""
    out = pd.DataFrame({e: run["returns"][strategy] for e, run in runs.items()})
    for k in ("ew", "vw"):
        out[k] = runs["single_factor"]["returns"][k]
    return out


def event_window(market: pd.Series, peak: int, trough: int) -> dict:
    """Crash and rebound months of one market drawdown.

    market : monthly market total returns indexed by yyyymm.
    peak   : last month at the market's previous high (the event starts after it).
    trough : month in which the market's drawdown bottoms out.

    The crash runs from the month after `peak` to `trough`; the rebound runs
    from the month after `trough` to the first month in which the market's
    wealth regains its level at `peak`. Both dates are checked against the data.
    """
    months = market.index
    wealth = (1.0 + market).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    if drawdown.loc[peak] != 0.0:
        raise ValueError(f"{peak} is not a market high")
    if drawdown.loc[peak:trough].idxmin() != trough:
        raise ValueError(f"{trough} is not the bottom of the drawdown after {peak}")

    after = wealth.loc[months[months.get_loc(trough) + 1]:]
    recovered = after.index[after >= wealth.loc[peak]]
    if len(recovered) == 0:
        raise ValueError(f"the market has not regained its {peak} level")
    return dict(peak=peak, trough=trough, recovery=int(recovered[0]),
                crash=(int(months[months.get_loc(peak) + 1]), trough),
                rebound=(int(months[months.get_loc(trough) + 1]), int(recovered[0])))


def event_study(returns: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Crash return, maximum drawdown during the crash, and rebound return.

    windows maps an event label to the output of `event_window`. Wealth is
    measured from 1 at the end of the peak month, so the drawdown includes any
    loss from that starting level.
    """
    rows = {}
    for label, w in windows.items():
        crash = returns.loc[w["crash"][0]:w["crash"][1]]
        wealth = (1.0 + crash).cumprod()
        rows[(label, "Crash return (%)")] = 100 * (wealth.iloc[-1] - 1.0)
        rows[(label, "Max drawdown (%)")] = 100 * (wealth / wealth.cummax().clip(lower=1.0) - 1.0).min()
        rebound = returns.loc[w["rebound"][0]:w["rebound"][1]]
        rows[(label, "Rebound return (%)")] = 100 * ((1.0 + rebound).prod() - 1.0)
    return pd.DataFrame(rows).T.rename_axis(["Event", "Metric"])


def plot_event_paths(returns: pd.DataFrame, windows: dict, colors=None, labels=None, linestyles=None):
    """One panel per event: wealth from 1 at the peak through the market's recovery,
    with the crash shaded."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    colors, labels, linestyles = colors or COLORS, labels or LABELS, linestyles or {}
    months = returns.index
    # Points are wealth at month end, so the peak point is 1 at the end of the peak month.
    month_end = lambda m: pd.PeriodIndex([str(d) for d in m], freq="M").to_timestamp(how="end")
    fig, axes = plt.subplots(1, len(windows), figsize=(5.2 * len(windows), 4.2))
    for ax, (label, w) in zip(np.atleast_1d(axes), windows.items()):
        span = months[months.get_loc(w["peak"]):months.get_loc(w["recovery"]) + 1]
        path = (1.0 + returns.loc[span[1:]]).cumprod()
        path = pd.concat([pd.DataFrame(1.0, index=span[:1], columns=returns.columns), path])
        when = month_end(span)
        crash = month_end([w["peak"], w["trough"]])
        ax.axvspan(crash[0], crash[1], color="0.9", zorder=0)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for k in returns.columns:
            ax.plot(when, path[k], color=colors.get(k), linestyle=linestyles.get(k, "-"),
                    label=labels.get(k, k))
        ax.axhline(1.0, color="0.5", linewidth=0.8)
        ax.set_title(f"{label}\npeak {w['peak']}, trough {w['trough']}, market recovery {w['recovery']}",
                     fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("wealth, 1 at the market peak")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig
