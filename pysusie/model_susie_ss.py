"""
SuSiE-SS (summary/sufficient-statistics) implementation aligned with model_SuSiE.py.

This module provides an Iterative Bayesian Stepwise Selection (IBSS) algorithm using
either sufficient statistics (XtX, Xty, yty, N) or summary statistics (z, R, N)
mapped to sufficient statistics. It mirrors the behavior and interfaces of model_SuSiE.py,
including prior variance optimization, credible set construction, and PIP computation.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from typing import Optional, Dict, Any, List, Tuple, Union

# Utilities shared with the SuSiE package API (minimal re-definitions)

def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Stable logsumexp along axis.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int or None
        Axis to reduce; if None, reduce over all elements.

    Returns
    -------
    ndarray
        Log-sum-exp with the specified axis removed (if axis is not None).
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def _logsumexp_full(lpo: np.ndarray) -> float:
    """Stable log(sum(exp(lpo))) for a 1D array."""
    m = np.max(lpo)
    return float(m + np.log(np.sum(np.exp(lpo - m))))

def is_symmetric_matrix(X: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if X is symmetric within absolute tolerance tol."""
    return np.allclose(X, X.T, atol=tol, rtol=0)

def muffled_corr(X: np.ndarray) -> np.ndarray:
    """Compute a correlation matrix robustly to zero-variance columns.

    Zero-variance columns yield zero correlations to others and unit diagonal.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1)
    zero = sd < 1e-15
    sd_safe = sd.copy(); sd_safe[zero] = 1.0
    Xn = Xc / sd_safe
    R = (Xn.T @ Xn) / max(n - 1, 1)
    R[zero, :] = 0
    R[:, zero] = 0
    np.fill_diagonal(R, 1.0)
    return R

def n_in_CS_x(x: np.ndarray, coverage: float = 0.9) -> int:
    """Minimal count of top probabilities in x needed to reach 'coverage'."""
    xs = np.sort(x)[::-1]
    csum = np.cumsum(xs)
    return int(np.sum(csum < coverage) + 1)

def in_CS_x(x: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    """0/1 indicator selecting top entries of x achieving the target coverage."""
    n = n_in_CS_x(x, coverage)
    o = np.argsort(x)[::-1]
    result = np.zeros_like(x, dtype=int)
    result[o[:n]] = 1
    return result

def in_CS(res: Union[Dict[str, Any], np.ndarray], coverage: float = 0.9) -> np.ndarray:
    """Credible set membership for all L effects as an (L, p) indicator matrix."""
    if isinstance(res, dict):
        alpha = res["alpha"]
    else:
        alpha = np.asarray(res)
    status = np.vstack([in_CS_x(alpha[l, :], coverage) for l in range(alpha.shape[0])])
    return status


def get_purity(pos: List[int], X: Optional[np.ndarray], Xcorr: Optional[np.ndarray], squared: bool = False,
               n: int = 100, use_rfast: Optional[bool] = None, rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Purity metrics (min/mean/median abs correlation) within a set of indices.

    Subsamples to at most n indices for efficiency on large sets. Uses either raw X to
    recompute correlations or a precomputed correlation matrix Xcorr.
    """
    pos = list(pos)
    if len(pos) == 1:
        return (1.0, 1.0, 1.0)
    if len(pos) > n:
        if rng is None:
            rng = np.random.default_rng(0)
        pos = list(rng.choice(pos, size=n, replace=False))
    if Xcorr is None:
        if X is None:
            return (np.nan, np.nan, np.nan)
        X_sub = np.asarray(X[:, pos])
        R = muffled_corr(X_sub)
    else:
        R = np.asarray(Xcorr)[np.ix_(pos, pos)]
        R = 0.5 * (R + R.T)
    idx = np.triu_indices(len(pos), k=1)
    vals = np.abs(R[idx])
    if squared:
        vals = vals**2
    if vals.size == 0:
        return (1.0, 1.0, 1.0)
    return (float(np.min(vals)), float(np.mean(vals)), float(np.median(vals)))


def susie_get_cs_py(res: Dict[str, Any], X: Optional[np.ndarray] = None, Xcorr: Optional[np.ndarray] = None,
                    coverage: float = 0.95, min_abs_corr: float = 0.5, dedup: bool = True, squared: bool = False,
                    check_symmetric: bool = True, n_purity: int = 100, use_rfast: Optional[bool] = None) -> Dict[str, Any]:
    """Construct credible sets (CS) from alpha and compute purity with X or Xcorr.

    Deduplicates identical CS across effects, filters by a minimum absolute correlation
    threshold, and returns CS indices, purity summaries, and claimed coverage.
    """
    if (X is not None) and (Xcorr is not None):
        raise ValueError("Only one of X or Xcorr should be specified")
    if (Xcorr is not None) and check_symmetric and (not is_symmetric_matrix(Xcorr)):
        Xcorr = 0.5 * (Xcorr + Xcorr.T)
    null_index = res.get("null_index", 0) or 0
    alpha = res["alpha"]
    V = res.get("V", None)
    if isinstance(V, np.ndarray):
        include_idx = (V > 1e-9)
    elif isinstance(V, (float, int)):
        include_idx = np.ones(alpha.shape[0], dtype=bool)
    else:
        include_idx = np.ones(alpha.shape[0], dtype=bool)
    status = in_CS(alpha, coverage)
    cs = []
    claimed_coverage = []
    for i in range(status.shape[0]):
        pos = np.where(status[i, :] != 0)[0].tolist()
        cs.append(pos)
        claimed_coverage.append(float(np.sum(alpha[i, pos])))
    include_idx = include_idx & (np.array([len(v) for v in cs]) > 0)
    if dedup:
        seen = set(); mask = []
        for i, sset in enumerate(cs):
            key = tuple(sorted(sset))
            if (key not in seen) and include_idx[i]:
                seen.add(key); mask.append(True)
            else:
                mask.append(False)
        include_idx = include_idx & np.array(mask)
    if np.sum(include_idx) == 0:
        return dict(cs=None, coverage=None, requested_coverage=coverage)
    cs = [cs[i] for i in np.where(include_idx)[0]]
    claimed_coverage = [claimed_coverage[i] for i in np.where(include_idx)[0]]
    if (Xcorr is None) and (X is None):
        cs_dict = {f"L{k}": [int(idx) for idx in sset] for k, sset in enumerate(cs)}
        coverage_map = {str(k): float(cov) for k, cov in enumerate(claimed_coverage)}
        return dict(cs=cs_dict, purity=None, cs_index=list(range(len(cs))), coverage=coverage_map,
                    requested_coverage=float(coverage))
    purity = []
    rng = np.random.default_rng(0)
    for sset in cs:
        if (null_index > 0) and (null_index in sset):
            purity.append((-9.0, -9.0, -9.0))
        else:
            purity.append(get_purity(sset, X, Xcorr, squared=squared, n=n_purity, rng=rng))
    purity = np.asarray(purity, float)
    threshold = (min_abs_corr**2) if squared else min_abs_corr
    is_pure = np.where(purity[:, 0] >= threshold)[0]
    if is_pure.size == 0:
        return dict(cs=None, coverage=None, requested_coverage=coverage)
    cs = [cs[i] for i in is_pure]
    purity = purity[is_pure, :]
    cs_index = list(range(len(cs)))
    covg = np.asarray(claimed_coverage, float)[is_pure]
    ordering = np.argsort(purity[:, 0])[::-1]
    cs = [cs[i] for i in ordering]
    purity = purity[ordering, :]
    cs_index = [int(cs_index[i]) for i in ordering]
    covg = covg[ordering]
    cs_dict = {f"L{k}": [int(idx) for idx in sset] for k, sset in enumerate(cs)}
    purity_dict = {
        "min_abs_corr": {f"L{k}": float(purity[k, 0]) for k in range(len(cs))},
        "mean_abs_corr": {f"L{k}": float(purity[k, 1]) for k in range(len(cs))},
        "median_abs_corr": {f"L{k}": float(purity[k, 2]) for k in range(len(cs))}
    }
    coverage_map = {str(k): float(covg[k]) for k in range(len(cs))}
    return dict(cs=cs_dict, purity=purity_dict, cs_index=[int(i) for i in cs_index], coverage=coverage_map,
                requested_coverage=float(coverage))


def susie_get_pip_py(res: Union[Dict[str, Any], np.ndarray], prune_by_cs: bool = False, prior_tol: float = 1e-9) -> np.ndarray:
    """Posterior inclusion probabilities (PIPs): 1 - prod_l (1 - alpha_lj)."""
    if isinstance(res, dict):
        alpha = res["alpha"]
        null_index = res.get("null_index", None)
        if null_index is not None and null_index > 0:
            alpha = np.delete(alpha, null_index, axis=1)
        V = res.get("V", None)
        if isinstance(V, np.ndarray) or isinstance(V, (float, int)):
            if np.ndim(V) == 0:
                include_idx = np.arange(alpha.shape[0])
            else:
                include_idx = np.where(np.asarray(V) > prior_tol)[0]
        else:
            include_idx = np.arange(alpha.shape[0])
        if prune_by_cs and res.get("sets", None) is not None and ("cs_index" in res["sets"]):
            include_idx = np.intersect1d(include_idx, np.asarray(res["sets"]["cs_index"], dtype=int))
        alpha_use = alpha[include_idx, :] if include_idx.size > 0 else np.zeros((1, alpha.shape[1]))
    else:
        alpha_use = np.asarray(res)
    alpha_use = np.nan_to_num(alpha_use, nan=0.0, posinf=1.0, neginf=0.0)
    return 1.0 - np.prod(1.0 - alpha_use, axis=0)


def _loglik_core(Vv: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """Core single-effect marginal log-likelihood across variables.

    Invalid columns (non-finite or non-positive shat2) contribute Bayes factor of 1.
    """
    lpo = np.log(prior + np.sqrt(np.finfo(float).eps))
    mask = np.isfinite(shat2) & (shat2 > 0)
    if np.any(mask):
        denom = Vv + shat2[mask]
        denom = np.maximum(denom, np.finfo(float).tiny)
        lbf_mask = ((-0.5 * np.log(2 * np.pi * denom) - 0.5 * (betahat[mask]**2) / denom) -
                    (-0.5 * np.log(2 * np.pi * shat2[mask]) - 0.5 * (betahat[mask]**2) / shat2[mask]))
        lpo[mask] = lpo[mask] + lbf_mask
    return _logsumexp_full(lpo)


def neg_loglik_logscale(lV: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """Negative log-likelihood with V parameterized on the log scale (lV=log V)."""
    return float(-_loglik_core(np.exp(lV), betahat, shat2, prior))


def loglik_grad(V: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """Gradient of the log-likelihood with respect to V (not log V)."""
    p = prior.size
    lpo = np.log(prior + np.sqrt(np.finfo(float).eps))
    mask = np.isfinite(shat2) & (shat2 > 0)
    grad_sum = 0.0
    if np.any(mask):
        denom = V + shat2[mask]
        denom = np.maximum(denom, np.finfo(float).tiny)
        lbf_mask = ((-0.5 * np.log(2 * np.pi * denom) - 0.5 * (betahat[mask]**2) / denom) -
                    (-0.5 * np.log(2 * np.pi * shat2[mask]) - 0.5 * (betahat[mask]**2) / shat2[mask]))
        lpo[mask] = lpo[mask] + lbf_mask
        m = np.max(lpo)
        w = np.exp(lpo - m)
        sw = np.sum(w)
        alpha_full = w / sw if sw > 0 else np.repeat(1.0 / p, p)
        T2 = (betahat[mask]**2) / shat2[mask]
        grad_vec = 0.5 * (1.0 / denom) * ((shat2[mask] / denom) * T2 - 1.0)
        grad_vec[np.isnan(grad_vec)] = 0.0
        grad_sum = float(np.sum(alpha_full[mask] * grad_vec))
    else:
        grad_sum = 0.0
    return grad_sum


def negloglik_grad_logscale(lV: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """Negative gradient of log-likelihood with respect to log V (chain rule)."""
    V = np.exp(lV)
    return float(-V * loglik_grad(V, betahat, shat2, prior))


def est_V_uniroot(betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """Estimate prior variance V by bracketing and Brent root-finding on gradient in log V."""
    def g(lV):
        return negloglik_grad_logscale(lV, betahat, shat2, prior)
    bounds = [(-10.0, 10.0), (-20.0, 20.0), (-30.0, 30.0)]
    for a, b in bounds:
        fa, fb = g(a), g(b)
        if np.isfinite(fa) and np.isfinite(fb) and (fa * fb <= 0):
            sol = root_scalar(g, bracket=(a, b), method="brentq", xtol=1e-8, rtol=1e-8, maxiter=200)
            if sol.converged:
                return float(np.exp(sol.root))
    return float(np.exp(0.5 * (bounds[-1][0] + bounds[-1][1])))


def optimize_prior_variance(optimize_V: str, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray,
                            alpha: Optional[np.ndarray] = None, post_mean2: Optional[np.ndarray] = None,
                            V_init: Optional[float] = None, check_null_threshold: float = 0.0) -> float:
    """Optimize or update the prior variance V via one of: 'optim', 'uniroot', 'EM', or 'none'.

    Applies a null-preferred check: if null model (V=0) is not worse than current V by
    at least check_null_threshold, set V=0.
    """
    V = float(V_init) if V_init is not None else 0.0
    if optimize_V != "simple":
        if optimize_V == "optim":
            def f(lV):
                return neg_loglik_logscale(lV, betahat, shat2, prior)
            res = minimize_scalar(f, bracket=(-30.0, 15.0), method="brent", options={"maxiter": 500})
            lV_new = float(res.x)
            if V_init is not None and np.isfinite(V_init) and (V_init > 0):
                if f(lV_new) > f(np.log(V_init)):
                    lV_new = np.log(V_init)
            V = float(np.exp(lV_new))
        elif optimize_V == "uniroot":
            V = est_V_uniroot(betahat, shat2, prior)
        elif optimize_V == "EM":
            if alpha is None or post_mean2 is None:
                raise ValueError("EM requires alpha and post_mean2")
            V = float(np.sum(alpha * post_mean2))
        elif optimize_V == "none":
            V = float(V)
        else:
            raise ValueError("Invalid option for optimize_V method")
    if _loglik_core(0.0, betahat, shat2, prior) + check_null_threshold >= _loglik_core(V, betahat, shat2, prior):
        V = 0.0
    return V


class SuSiE_SS:
    """
    SuSiE-SS: IBSS using sufficient statistics (XtX, Xty, yty, N) or summary data (R, z, N).

    Naming aligned with model_SuSiE.py:
      - State dict s fields: alpha, mu, mu2, sigma2, V, pi, KL, lbf, lbf_variable, null_index.
      - Class attributes: alpha, mu, mu2, sigma2, V, pip, sets, elbo, niter, converged.

    The fit method supports:
      - sufficient=True: direct sufficient statistics input.
      - sufficient=False: summary statistics mapping to sufficient statistics.
    """
    def __init__(self,
                 L: int = 10,
                 scaled_prior_variance: float = 0.2,
                 estimate_prior_variance: bool = True,
                 estimate_prior_method: str = "optim",
                 check_null_threshold: float = 0.0,
                 estimate_residual_variance: bool = True,
                 tol: float = 1e-6,
                 max_iter: int = 200,
                 verbose: bool = False,
                 seed: int = 1):
        self.L = int(L)
        self.scaled_prior_variance = float(scaled_prior_variance)
        self.estimate_prior_variance = bool(estimate_prior_variance)
        self.estimate_prior_method = str(estimate_prior_method)
        self.check_null_threshold = float(check_null_threshold)
        self.estimate_residual_variance = bool(estimate_residual_variance)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(seed)
        self.alpha: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.mu2: Optional[np.ndarray] = None
        self.sigma2: Optional[float] = None
        self.V: Optional[np.ndarray] = None
        self.pip: Optional[np.ndarray] = None
        self.sets: Optional[Dict[str, Any]] = None
        self.elbo: Optional[np.ndarray] = None
        self.niter: int = 0
        self.converged: bool = False
        self.R_ss: Optional[np.ndarray] = None

    @staticmethod
    def _pve_adjusted_z(z: np.ndarray, N: int) -> np.ndarray:
        """PVE attenuation for z-scores to reduce winner's curse bias."""
        Dz = N / (N + np.square(z))
        return z * np.sqrt(Dz)

    @staticmethod
    def _symmetrize_corr(R: np.ndarray) -> np.ndarray:
        """Symmetrize correlation matrix and set diagonal to 1."""
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
        return R

    def _build_sufficient_from_summary(self, z: np.ndarray, R: np.ndarray, N: int,
                                       use_pve_adjust: bool = True) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Map (z, R, N) summary inputs to sufficient statistics (XtX, Xty, yty, N)."""
        R = self._symmetrize_corr(np.asarray(R, dtype=np.float64))
        z = np.asarray(z, dtype=np.float64).ravel()
        p = z.size
        if R.shape != (p, p):
            raise ValueError(f"R shape {R.shape} incompatible with z length {p}.")
        z_tilde = self._pve_adjusted_z(z, N) if use_pve_adjust else z.copy()
        XtX = N * R
        Xty = np.sqrt(N) * z_tilde
        yty = float(N)
        return XtX, Xty, yty, int(N)

    @staticmethod
    def _init_state(p: int, L: int, varY: float, scaled_prior_variance: float) -> Dict[str, Any]:
        """Initialize state dictionary for SuSiE-SS with shapes aligned to (L, p)."""
        s = dict(
            alpha=np.full((L, p), 1.0 / p, dtype=np.float64),
            mu=np.zeros((L, p), dtype=np.float64),
            mu2=np.zeros((L, p), dtype=np.float64),
            KL=np.full(L, np.nan),
            lbf=np.full(L, np.nan),
            lbf_variable=np.full((L, p), np.nan),
            sigma2=float(varY),
            V=np.full(L, scaled_prior_variance * float(varY), dtype=np.float64),
            pi=np.repeat(1.0 / p, p),
            null_index=0
        )
        return s

    @staticmethod
    def _elbo_ss(XtX: np.ndarray, Xty: np.ndarray, yty: float, N: int,
                 alpha: np.ndarray, mu: np.ndarray, mu2: np.ndarray, sigma2: float) -> float:
        """ELBO-like objective under sufficient statistics with diagonal variance approx."""
        diagXtX = np.diag(XtX)
        beta = np.sum(alpha * mu, axis=0)
        EbTXb = beta @ XtX @ beta
        var_terms = np.sum((alpha * (mu2 - np.square(mu))) * diagXtX, axis=1)
        EbTXb += float(np.sum(var_terms))
        sse = yty - 2.0 * (beta @ Xty) + EbTXb
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -np.nansum(alpha * np.log(alpha + 1e-300))
        elbo = -0.5 * N * np.log(2.0 * np.pi * sigma2) - 0.5 * sse / sigma2 + ent / max(alpha.shape[0], 1)
        return float(elbo)

    def fit(self,
               z: Optional[np.ndarray] = None,
               R: Optional[np.ndarray] = None,
               N: Optional[int] = None,
               XtX: Optional[np.ndarray] = None,
               Xty: Optional[np.ndarray] = None,
               yty: Optional[float] = None,
               sufficient: bool = False,
               use_pve_adjust: bool = True,
               sigma2: Optional[float] = None,
               coverage: float = 0.95,
               min_abs_corr: float = 0.5,
               n_purity: int = 100) -> "SuSiE_SS":
        """Fit SuSiE-SS using sufficient stats or (z, R, N) summary data.

        When sufficient=False, z and R are converted to XtX, Xty, yty using a PVE-adjusted
        mapping. Returns self with attributes alpha, mu, mu2, sigma2, V, pip, sets, elbo.
        """
        if sufficient:
            if XtX is None or Xty is None or yty is None or N is None:
                raise ValueError("sufficient=True requires XtX, Xty, yty, N.")
            XtX = np.asarray(XtX, dtype=np.float64)
            Xty = np.asarray(Xty, dtype=np.float64).ravel()
            yty = float(yty)
            N = int(N)
            p = XtX.shape[0]
            if XtX.shape != (p, p) or Xty.shape[0] != p:
                raise ValueError("Shapes of XtX/Xty are inconsistent.")
            self.R_ss = None
        else:
            if z is None or R is None or N is None:
                raise ValueError("sufficient=False requires z, R, N.")
            XtX, Xty, yty, N = self._build_sufficient_from_summary(z, R, int(N), use_pve_adjust=use_pve_adjust)
            p = XtX.shape[0]
            self.R_ss = self._symmetrize_corr(np.asarray(R, float))
        XtX = 0.5 * (XtX + XtX.T)
        diagXtX = np.diag(XtX)
        if np.any(diagXtX <= 0):
            raise ValueError("XtX has non-positive diagonal entries.")
        varY = yty / N if N > 0 else 1.0
        s = self._init_state(p=p, L=self.L, varY=varY, scaled_prior_variance=self.scaled_prior_variance)
        if sigma2 is not None:
            s["sigma2"] = float(sigma2)
            do_update_sigma2 = False
        else:
            do_update_sigma2 = bool(self.estimate_residual_variance)
        elbo = np.full(self.max_iter + 1, np.nan, float)
        elbo[0] = -np.inf
        it = 0  # initialize loop counter for static analyzers
        for it in range(1, self.max_iter + 1):
            beta_all = np.sum(s["alpha"] * s["mu"], axis=0)
            for l in range(self.L):
                beta_minus_l = beta_all - s["alpha"][l, :] * s["mu"][l, :]
                rX = Xty - XtX @ beta_minus_l
                bhat = rX / np.maximum(diagXtX, 1e-30)
                s2j = s["sigma2"] / np.maximum(diagXtX, 1e-30)
                if self.estimate_prior_variance and self.estimate_prior_method in ("optim", "uniroot"):
                    s["V"][l] = optimize_prior_variance(
                        optimize_V=self.estimate_prior_method,
                        betahat=bhat,
                        shat2=s2j,
                        prior=s["pi"],
                        alpha=None,
                        post_mean2=None,
                        V_init=float(s["V"][l]),
                        check_null_threshold=self.check_null_threshold
                    )
                V_l = float(s["V"][l])
                denom = np.maximum(s2j + V_l, np.finfo(float).tiny)
                lbf = 0.5 * (np.log(np.maximum(s2j, 1e-300)) - np.log(denom)) \
                      + 0.5 * (bhat**2) * (V_l / np.maximum(s2j * denom, 1e-300))
                s["lbf_variable"][l, :] = lbf
                log_pi = np.log(s["pi"] + 1e-300)
                lpo = log_pi + lbf
                m = np.max(lpo)
                w = np.exp(lpo - m)
                sw = np.sum(w)
                alpha_l = w / sw if sw > 0 else np.repeat(1.0 / p, p)
                s["alpha"][l, :] = alpha_l
                if V_l > 0:
                    shrink = V_l / np.maximum(V_l + s2j, 1e-300)
                    var_post = shrink * s2j
                    mean_post = shrink * bhat
                else:
                    var_post = np.zeros(p)
                    mean_post = np.zeros(p)
                s["mu"][l, :] = mean_post
                s["mu2"][l, :] = mean_post**2 + var_post
                s["lbf"][l] = float(m + np.log(sw)) if sw > 0 else -np.inf
                if self.estimate_prior_variance and (self.estimate_prior_method == "EM"):
                    V_new = optimize_prior_variance(
                        optimize_V="EM",
                        betahat=bhat,
                        shat2=s2j,
                        prior=s["pi"],
                        alpha=s["alpha"][l, :],
                        post_mean2=s["mu2"][l, :],
                        V_init=float(s["V"][l]),
                        check_null_threshold=self.check_null_threshold
                    )
                    s["V"][l] = float(V_new)
                beta_all = beta_minus_l + s["alpha"][l, :] * s["mu"][l, :]
            if do_update_sigma2:
                beta = beta_all
                EbTXb = beta @ XtX @ beta
                var_terms = np.sum((s["alpha"] * (s["mu2"] - np.square(s["mu"]))) * diagXtX, axis=1)
                EbTXb += float(np.sum(var_terms))
                sse = yty - 2.0 * (beta @ Xty) + EbTXb
                sigma2_new = max(sse / N, 1e-16)
            else:
                sigma2_new = s["sigma2"]
            elbo[it] = self._elbo_ss(XtX, Xty, yty, N, s["alpha"], s["mu"], s["mu2"], sigma2_new)
            if self.verbose:
                print(f"[SuSiE-SS] iter={it:03d} sigma2={sigma2_new:.6g} ELBO={elbo[it]:.6f}")
            if (elbo[it] - elbo[it - 1]) < self.tol:
                s["converged"] = True
                s["sigma2"] = sigma2_new
                break
            s["sigma2"] = sigma2_new
        used_iter = min(it, self.max_iter)
        self.elbo = elbo[1:used_iter + 1]
        self.niter = used_iter
        self.converged = bool(s.get("converged", False))
        self.alpha = s["alpha"]
        self.mu = s["mu"]
        self.mu2 = s["mu2"]
        self.sigma2 = float(s["sigma2"])
        self.V = s["V"]
        xcorr = self.R_ss if (self.R_ss is not None) else None
        s_for_cs = dict(alpha=self.alpha, V=self.V, null_index=s.get("null_index", 0))
        self.sets = susie_get_cs_py(res=s_for_cs, X=None, Xcorr=xcorr, coverage=coverage, min_abs_corr=min_abs_corr,
                                    dedup=True, squared=False, check_symmetric=True, n_purity=n_purity)
        self.pip = susie_get_pip_py(s_for_cs, prune_by_cs=False, prior_tol=1e-9)
        return self
