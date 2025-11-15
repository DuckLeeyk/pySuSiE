"""
SuSiE model implementation using NumPy and SciPy.

This module implements a NumPy/Scipy-based version of the SuSiE (Sum of Single Effects) model
for sparse regression and fine-mapping. It includes utilities for column statistics, correlation,
credible set construction, posterior inclusion probabilities (PIPs), and auxiliary univariate regression helpers.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from typing import Optional, Dict, Any, List, Tuple, Union

# Utility linear algebra helpers

def compute_Xb(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute X @ b.

    Parameters
    ----------
    X : ndarray (n, p)
    b : ndarray (p,)
    Returns
    -------
    ndarray (n,) result of matrix-vector product.
    """
    return X @ b

def compute_MXt(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute (X @ M.T).T.

    Each row l of the returned matrix equals X @ M[l,].
    """
    return (X @ M.T).T

def compute_Xty(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute X.T @ y cross-product vector."""
    return X.T @ y

def compute_colSds(X: np.ndarray) -> np.ndarray:
    """Column standard deviations (unbiased, ddof=1) with rounding safeguard."""
    X = np.asarray(X, float)
    n = X.shape[0]
    col_mean = np.mean(X, axis=0)
    col_mean_sq = np.mean(X**2, axis=0)
    var = (col_mean_sq - col_mean**2) * (n / max(n - 1, 1))
    var = np.maximum(var, 0.0)
    return np.sqrt(var)

def compute_colstats(X: np.ndarray, center: bool = True, scale: bool = True) -> Dict[str, np.ndarray]:
    """Return column statistics: means (cm), sds (csd), standardized squared norms (d)."""
    X = np.asarray(X, float)
    n, p = X.shape
    cm = np.mean(X, axis=0) if center else np.zeros(p, float)
    if scale:
        csd = compute_colSds(X); csd[csd == 0] = 1.0
    else:
        csd = np.ones(p, float)
    col_mean = np.mean(X, axis=0)
    d_raw = n * (col_mean**2) + (n - 1) * (compute_colSds(X)**2)
    d = (d_raw - n * (cm**2)) / (csd**2)
    return dict(cm=cm, csd=csd, d=d)

def muffled_corr(X: np.ndarray) -> np.ndarray:
    """Safe correlation matrix: zero-variance columns get 0 correlations (diagonal=1)."""
    X = np.asarray(X, float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1)
    zero = sd < 1e-15
    sd_safe = sd.copy(); sd_safe[zero] = 1.0
    Xn = Xc / sd_safe
    R = (Xn.T @ Xn) / (n - 1)
    R[zero, :] = 0; R[:, zero] = 0
    np.fill_diagonal(R, 1.0)
    return R

def is_symmetric_matrix(X: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if |X - X.T|_{max} <= tol."""
    return np.allclose(X, X.T, atol=tol, rtol=0)

def n_in_CS_x(x: np.ndarray, coverage: float = 0.9) -> int:
    """Minimal number of largest elements of x whose cumulative sum reaches coverage."""
    xs = np.sort(x)[::-1]
    csum = np.cumsum(xs)
    return int(np.sum(csum < coverage) + 1)

def in_CS_x(x: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    """Return 0/1 indicator selecting top entries of x achieving target coverage."""
    n = n_in_CS_x(x, coverage)
    o = np.argsort(x)[::-1]
    result = np.zeros_like(x, dtype=int)
    result[o[:n]] = 1
    return result

def in_CS(res: Union[Dict[str, Any], np.ndarray], coverage: float = 0.9) -> np.ndarray:
    if isinstance(res, dict):
        alpha = res["alpha"]
    else:
        alpha = np.asarray(res)
    return np.vstack([in_CS_x(alpha[l, :], coverage) for l in range(alpha.shape[0])])

def get_purity(pos: List[int], X: Optional[np.ndarray], Xcorr: Optional[np.ndarray], squared: bool = False,
               n: int = 100, use_rfast: Optional[bool] = None, rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Purity metrics (min/mean/median abs correlation) within index set pos (subsampled if large)."""
    pos = list(pos)
    if len(pos) == 1: return (1.0, 1.0, 1.0)
    if len(pos) > n:
        if rng is None: rng = np.random.default_rng(0)
        pos = list(rng.choice(pos, size=n, replace=False))
    if Xcorr is None:
        X_sub = np.asarray(X[:, pos]); R = muffled_corr(X_sub)
    else:
        R = np.asarray(Xcorr)[np.ix_(pos, pos)]; R = 0.5 * (R + R.T)
    idx = np.triu_indices(len(pos), k=1); vals = np.abs(R[idx])
    if squared: vals = vals**2
    if vals.size == 0: return (1.0, 1.0, 1.0)
    return (float(np.min(vals)), float(np.mean(vals)), float(np.median(vals)))

def susie_get_cs_py(res: Dict[str, Any], X: Optional[np.ndarray] = None, Xcorr: Optional[np.ndarray] = None,
                    coverage: float = 0.95, min_abs_corr: float = 0.5, dedup: bool = True, squared: bool = False,
                    check_symmetric: bool = True, n_purity: int = 100, use_rfast: Optional[bool] = None) -> Dict[str, Any]:
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
    cs = []; claimed_coverage = []
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
    purity = []; rng = np.random.default_rng(0)
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
    cs = [cs[i] for i in is_pure]; purity = purity[is_pure, :]
    cs_index = list(range(len(cs))); covg = np.asarray(claimed_coverage, float)[is_pure]
    ordering = np.argsort(purity[:, 0])[::-1]
    cs = [cs[i] for i in ordering]; purity = purity[ordering, :]
    cs_index = [int(cs_index[i]) for i in ordering]; covg = covg[ordering]
    cs_dict = {f"L{k}": [int(idx) for idx in sset] for k, sset in enumerate(cs)}
    purity_dict = {"min_abs_corr": {f"L{k}": float(purity[k, 0]) for k in range(len(cs))},
                   "mean_abs_corr": {f"L{k}": float(purity[k, 1]) for k in range(len(cs))},
                   "median_abs_corr": {f"L{k}": float(purity[k, 2]) for k in range(len(cs))}}
    coverage_map = {str(k): float(covg[k]) for k in range(len(cs))}
    return dict(cs=cs_dict, purity=purity_dict, cs_index=[int(i) for i in cs_index], coverage=coverage_map,
                requested_coverage=float(coverage))

def susie_get_pip_py(res: Union[Dict[str, Any], np.ndarray], prune_by_cs: bool = False, prior_tol: float = 1e-9) -> np.ndarray:
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

# Initialization helpers

def init_setup(n: int, p: int, L: int, scaled_prior_variance, residual_variance,
               prior_weights, null_weight, varY: float, standardize: bool) -> Dict[str, Any]:
    spv = np.asarray(scaled_prior_variance, float)
    if np.any(spv < 0):
        raise ValueError("Scaled prior variance should be positive")
    if np.any(spv > 1) and standardize:
        raise ValueError("Scaled prior variance should be <= 1 when standardize is True")
    if residual_variance is None:
        residual_variance = varY
    if prior_weights is None:
        prior_weights = np.repeat(1.0 / p, p)
    else:
        prior_weights = np.asarray(prior_weights, float)
        if np.all(prior_weights == 0):
            raise ValueError("Prior weight must be > 0 for at least one variable.")
        prior_weights = prior_weights / np.sum(prior_weights)
    if len(prior_weights) != p:
        raise ValueError("Prior weights must have length p")
    if p < L:
        L = p
    s = dict(alpha=np.full((L, p), 1.0 / p), mu=np.zeros((L, p)), mu2=np.zeros((L, p)), Xr=np.zeros(n),
             KL=np.full(L, np.nan), lbf=np.full(L, np.nan), lbf_variable=np.full((L, p), np.nan),
             sigma2=float(residual_variance), V=spv * float(varY), pi=prior_weights)
    if null_weight is None:
        s["null_index"] = 0
    else:
        s["null_index"] = p - 1
    return s

def init_finalize(s: Dict[str, Any], X: Optional[np.ndarray] = None, Xr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    s = dict(s)
    if np.ndim(s["V"]) == 0:
        s["V"] = np.repeat(float(s["V"]), s["alpha"].shape[0])
    s["sigma2"] = float(s["sigma2"])
    if s["sigma2"] <= 0:
        raise ValueError("Residual variance sigma2 must be positive")
    if not np.all(np.asarray(s["V"]) >= 0):
        raise ValueError("Prior variance must be non-negative")
    if s["mu"].shape != s["mu2"].shape or s["mu"].shape != s["alpha"].shape:
        raise ValueError("Dimension mismatch among mu, mu2 and alpha")
    if s["alpha"].shape[0] != len(s["V"]):
        raise ValueError("Length of V must equal nrow(alpha)")
    if Xr is not None:
        s["Xr"] = Xr
    if X is not None:
        s["Xr"] = compute_Xb(X, np.sum(s["mu"] * s["alpha"], axis=0))
    s["KL"] = np.full(s["alpha"].shape[0], np.nan)
    s["lbf"] = np.full(s["alpha"].shape[0], np.nan)
    return s

# ELBO utilities

def get_ER2(X: np.ndarray, Y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    Xr_L = compute_MXt(s["alpha"] * s["mu"], X)
    postb2 = s["alpha"] * s["mu2"]
    return float(np.sum((Y - s["Xr"])**2) - np.sum(Xr_L**2) + np.sum(d_attr * np.sum(postb2, axis=0)))

def SER_posterior_e_loglik(X: np.ndarray, Y: np.ndarray, s2: float, Eb: np.ndarray, Eb2: np.ndarray,
                           d_attr: np.ndarray) -> float:
    n = X.shape[0]
    term = -0.5 * n * np.log(2 * np.pi * s2) - 0.5 / s2 * (np.sum(Y * Y) - 2 * np.sum(Y * compute_Xb(X, Eb)) + np.sum(d_attr * Eb2))
    return float(term)

def Eloglik(X: np.ndarray, Y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    n = X.shape[0]
    return float(-(n / 2) * np.log(2 * np.pi * s["sigma2"]) - (1.0 / (2 * s["sigma2"])) * get_ER2(X, Y, s, d_attr=d_attr))

def get_objective(X: np.ndarray, Y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    return float(Eloglik(X, Y, s, d_attr=d_attr) - np.nansum(s["KL"]))

def _logsumexp_full(lpo: np.ndarray) -> float:
    m = np.max(lpo)
    return float(m + np.log(np.sum(np.exp(lpo - m))))

def _loglik_core(Vv: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
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
    return float(-_loglik_core(np.exp(lV), betahat, shat2, prior))

def loglik_grad(V: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
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
    return grad_sum

def negloglik_grad_logscale(lV: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    V = np.exp(lV); return float(-V * loglik_grad(V, betahat, shat2, prior))

def est_V_uniroot(betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    def g(lV): return negloglik_grad_logscale(lV, betahat, shat2, prior)
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
    V = float(V_init) if V_init is not None else 0.0
    if optimize_V != "simple":
        if optimize_V == "optim":
            def f(lV): return neg_loglik_logscale(lV, betahat, shat2, prior)
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

def single_effect_regression(y: np.ndarray, X: np.ndarray, V: float, residual_variance: float = 1.0,
                             prior_weights: Optional[np.ndarray] = None, optimize_V: str = "none",
                             check_null_threshold: float = 0.0, d_attr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    n, p = X.shape
    if d_attr is None:
        raise ValueError("single_effect_regression requires d_attr")
    Xty = compute_Xty(X, y); d = np.asarray(d_attr, float)
    betahat = np.zeros(p); shat2 = np.full(p, np.inf)
    good_d = np.isfinite(d) & (d > 0)
    if np.any(good_d):
        betahat[good_d] = Xty[good_d] / d[good_d]
        if residual_variance is not None and np.isfinite(residual_variance) and residual_variance > 0:
            shat2[good_d] = residual_variance / d[good_d]
    if prior_weights is None:
        prior = np.repeat(1.0 / p, p)
    else:
        prior = np.asarray(prior_weights, float); prior = np.maximum(prior, 0); sw = prior.sum()
        prior = prior / sw if sw > 0 else np.repeat(1.0 / p, p)
    if optimize_V not in ("EM", "none"):
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior, alpha=None, post_mean2=None,
                                    V_init=V, check_null_threshold=check_null_threshold)
    lbf = np.zeros(p); lpo = np.log(prior + np.sqrt(np.finfo(float).eps))
    mask = np.isfinite(shat2) & (shat2 > 0)
    if np.any(mask):
        denom = V + shat2[mask]; denom = np.maximum(denom, np.finfo(float).tiny)
        lbf_mask = ((-0.5 * np.log(2 * np.pi * denom) - 0.5 * (betahat[mask]**2) / denom) -
                    (-0.5 * np.log(2 * np.pi * shat2[mask]) - 0.5 * (betahat[mask]**2) / shat2[mask]))
        lbf[mask] = lbf_mask; lpo[mask] = lpo[mask] + lbf_mask
    m = np.max(lpo); w = np.exp(lpo - m); sw = np.sum(w)
    alpha = w / sw if sw > 0 else np.repeat(1.0 / p, p)
    post_var = np.zeros(p); post_mean = np.zeros(p)
    if V is not None and np.isfinite(V) and V > 0 and residual_variance is not None and residual_variance > 0 and np.isfinite(residual_variance):
        denom_all = (1.0 / V) + (d / residual_variance); ok = np.isfinite(denom_all) & (denom_all > 0)
        post_var[ok] = 1.0 / denom_all[ok]; post_mean[ok] = (Xty[ok] / residual_variance) * post_var[ok]
    post_mean2 = post_var + post_mean**2
    lbf_model = float(m + np.log(sw)) if sw > 0 else -np.inf
    sigma2_safe = residual_variance if (residual_variance is not None and np.isfinite(residual_variance) and residual_variance > 0) else 1.0
    loglik = float(lbf_model + np.sum(-0.5 * np.log(2 * np.pi * sigma2_safe) - 0.5 * (y**2) / sigma2_safe))
    if optimize_V == "EM":
        V = optimize_prior_variance("EM", betahat, shat2, prior, alpha=alpha, post_mean2=post_mean2,
                                    V_init=V, check_null_threshold=check_null_threshold)
    return dict(alpha=alpha, mu=post_mean, mu2=post_mean2, lbf=lbf, lbf_model=lbf_model, V=V, loglik=loglik)

def estimate_residual_variance_fn(X: np.ndarray, y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    n = X.shape[0]; ER2 = get_ER2(X, y, s, d_attr=d_attr); return float((1.0 / n) * ER2)

class SuSiE:
    def __init__(self, L: int = 10):
        self.L = int(L)
        self.alpha: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.mu2: Optional[np.ndarray] = None
        self.Xr: Optional[np.ndarray] = None
        self.lbf: Optional[np.ndarray] = None
        self.lbf_variable: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.sigma2: float = np.nan
        self.V: Optional[Union[np.ndarray, float]] = None
        self.elbo: Optional[np.ndarray] = None
        self.fitted: Optional[np.ndarray] = None
        self.sets: Optional[Dict[str, Any]] = None
        self.pip: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.niter: int = 0
        self.converged: bool = False
        self.X_column_scale_factors: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None
        self.null_index: Optional[int] = None
    def fit(self, X: np.ndarray, y: np.ndarray, scaled_prior_variance: Union[float, np.ndarray] = 0.2,
            residual_variance: Optional[float] = None, prior_weights: Optional[np.ndarray] = None, null_weight: float = 0.0,
            standardize: bool = True, intercept: bool = True, estimate_residual_variance: bool = True,
            estimate_prior_variance: bool = True, estimate_prior_method: str = "optim", check_null_threshold: float = 0.0,
            prior_tol: float = 1e-9, residual_variance_upperbound: float = np.inf,
            coverage: Optional[float] = 0.95, min_abs_corr: Optional[float] = 0.5, compute_univariate_zscore: bool = False,
            max_iter: int = 100, tol: float = 1e-3, verbose: bool = False, residual_variance_lowerbound: Optional[float] = None,
            n_purity: int = 100) -> "SuSiE":
        """Iterative Bayesian Stepwise Selection (IBSS) loop for SuSiE.

        Performs coordinate ascent over L single-effect components updating variational
        parameters and (optionally) residual/prior variances. Populates credible sets and PIPs.
        """
        X = np.asarray(X, float); y = np.asarray(y, float)
        if not np.all(np.isfinite(X)): raise ValueError("X contains non-finite values (NaN/Inf).")
        if not np.all(np.isfinite(y)): raise ValueError("y contains non-finite values (NaN/Inf).")
        n, p0 = X.shape
        null_used = (isinstance(null_weight, (int, float)) and (null_weight is not None) and (null_weight != 0))
        if null_used:
            if prior_weights is None:
                pw = np.repeat(1.0 / p0, p0)
            else:
                pw = np.asarray(prior_weights, float); pw = pw / np.sum(pw)
            if null_weight < 0 or null_weight >= 1: raise ValueError("null_weight must be in [0,1).")
            prior_weights = np.concatenate([pw * (1.0 - null_weight), np.array([null_weight])])
            X = np.column_stack([X, np.zeros(n)])
        p = X.shape[1]
        if self.L is None: self.L = min(10, p)
        else: self.L = min(self.L, p)
        mean_y = float(np.mean(y)); y_cent = y.copy()
        if intercept: y_cent -= mean_y
        colstats = compute_colstats(X, center=intercept, scale=standardize)
        cm, csd, dvec = colstats["cm"], colstats["csd"], colstats["d"]
        if residual_variance_lowerbound is None:
            residual_variance_lowerbound = (np.var(y_cent, ddof=1) / 1e4) if y_cent.size > 1 else 0.0
        s = init_setup(n=n, p=p, L=self.L, scaled_prior_variance=scaled_prior_variance, residual_variance=residual_variance,
                       prior_weights=prior_weights, null_weight=(None if not null_used else null_weight),
                       varY=float(np.var(y_cent, ddof=1)) if y_cent.size > 1 else 0.0, standardize=standardize)
        s = init_finalize(s, X=X)
        self.pi = s["pi"]; self.null_index = s.get("null_index", 0)
        elbo = np.full(max_iter + 1, np.nan, float); elbo[0] = -np.inf
        i = 0  # initialize loop counter for static analyzers
        for i in range(1, max_iter + 1):
            L_cur = s["alpha"].shape[0]
            for l in range(L_cur):
                s["Xr"] = s["Xr"] - compute_Xb(X, s["alpha"][l, :] * s["mu"][l, :])
                R = y_cent - s["Xr"]
                res = single_effect_regression(y=R, X=X, V=s["V"][l], residual_variance=s["sigma2"], prior_weights=s["pi"],
                                                optimize_V=(estimate_prior_method if estimate_prior_variance else "none"),
                                                check_null_threshold=check_null_threshold, d_attr=dvec)
                s["mu"][l, :] = res["mu"]; s["alpha"][l, :] = res["alpha"]; s["mu2"][l, :] = res["mu2"]
                s["V"][l] = res["V"]; s["lbf"][l] = res["lbf_model"]; s["lbf_variable"][l, :] = res["lbf"]
                Eb = res["alpha"] * res["mu"]; Eb2 = res["alpha"] * res["mu2"]
                s["KL"][l] = -res["loglik"] + SER_posterior_e_loglik(X, R, s["sigma2"], Eb, Eb2, d_attr=dvec)
                s["Xr"] = s["Xr"] + compute_Xb(X, s["alpha"][l, :] * s["mu"][l, :])
            elbo[i] = get_objective(X, y_cent, s, d_attr=dvec)
            if (elbo[i] - elbo[i - 1]) < tol:
                s["converged"] = True; break
            if estimate_residual_variance:
                sig2 = estimate_residual_variance_fn(X, y_cent, s, d_attr=dvec)
                sig2 = max(residual_variance_lowerbound, sig2)
                if sig2 > residual_variance_upperbound: sig2 = residual_variance_upperbound
                s["sigma2"] = sig2
                if verbose:
                    obj_now = get_objective(X, y_cent, s, d_attr=dvec)
                    print(f"objective: {obj_now:.6g}")
        used_iter = min(i, max_iter)
        self.elbo = elbo[1:used_iter + 1]; self.niter = used_iter; self.converged = bool(s.get("converged", False))
        if intercept:
            post_mean_all = np.sum(s["alpha"] * s["mu"], axis=0)
            intercept_hat = mean_y - np.sum(cm * (post_mean_all / csd))
            s["intercept"] = float(intercept_hat); s["fitted"] = s["Xr"] + mean_y
        else:
            s["intercept"] = 0.0; s["fitted"] = s["Xr"]
        if (coverage is not None) and (min_abs_corr is not None):
            s["sets"] = susie_get_cs_py(res=s, X=X, Xcorr=None, coverage=coverage, min_abs_corr=min_abs_corr, dedup=True,
                                         squared=False, check_symmetric=True, n_purity=n_purity)
            s["pip"] = susie_get_pip_py(s, prune_by_cs=False, prior_tol=prior_tol)
            if null_used and s.get("null_index", 0) >= 0 and s["pip"] is not None:
                s["pip"] = s["pip"][:p-1]
        if compute_univariate_zscore:
            s["z"] = calc_z(X[:, :p-1], y_cent, center=intercept, scale=standardize) if null_used \
                     else calc_z(X, y_cent, center=intercept, scale=standardize)
        s["X_column_scale_factors"] = csd
        self.alpha = s["alpha"]; self.mu = s["mu"]; self.mu2 = s["mu2"]; self.Xr = s["Xr"]
        self.lbf = s["lbf"]; self.lbf_variable = s["lbf_variable"]; self.intercept = float(s["intercept"])  # noqa
        self.sigma2 = float(s["sigma2"]); self.V = s["V"]; self.fitted = s["fitted"]; self.sets = s.get("sets", None)
        self.pip = s.get("pip", None); self.z = s.get("z", None); self.X_column_scale_factors = s["X_column_scale_factors"]
        self.null_index = s.get("null_index", 0); self.pi = s.get("pi", None)
        return self

# Univariate helpers

def univariate_regression(X: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray] = None, center: bool = True,
                          scale: bool = False, return_residuals: bool = False) -> Dict[str, Any]:
    X = np.asarray(X, float); y = np.asarray(y, float).copy(); mask = ~np.isnan(y)
    if not np.all(mask): X = X[mask, :]; y = y[mask]
    if center:
        y = y - y.mean()
        if scale:
            X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0, ddof=1), 1e-15)
        else:
            X = X - X.mean(axis=0)
    else:
        if scale:
            X = X / np.maximum(X.std(axis=0, ddof=1), 1e-15)
    X = np.nan_to_num(X, nan=0.0)
    if Z is not None:
        Z = np.asarray(Z, float)
        if center:
            if scale:
                Z = (Z - Z.mean(axis=0)) / np.maximum(Z.std(axis=0, ddof=1), 1e-15)
            else:
                Z = Z - Z.mean(axis=0)
        q, _ = np.linalg.qr(Z, mode='reduced'); y = y - q @ (q.T @ y)
    n, p = X.shape; betahat = np.zeros(p); sebetahat = np.zeros(p)
    for j in range(p):
        x = X[:, j]; Xj = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(Xj, y, rcond=None)
        beta_j = coef[1]; resid = y - Xj @ coef; sig2 = np.sum(resid**2) / max(n - 2, 1)
        XtX_inv = np.linalg.inv(Xj.T @ Xj); se_j = np.sqrt(sig2 * XtX_inv[1, 1])
        betahat[j] = beta_j; sebetahat[j] = se_j
    out = dict(betahat=betahat, sebetahat=sebetahat)
    if return_residuals and Z is not None: out["residuals"] = y
    return out

def calc_z(X: np.ndarray, Y: np.ndarray, center: bool = False, scale: bool = False) -> np.ndarray:
    def univariate_z(X_, Y_, center_, scale_):
        out = univariate_regression(X_, Y_, center=center_, scale=scale_)
        return out["betahat"] / out["sebetahat"]
    if Y.ndim == 1:
        return univariate_z(X, Y, center, scale)
    else:
        return np.column_stack([univariate_z(X, Y[:, i], center, scale) for i in range(Y.shape[1])])
