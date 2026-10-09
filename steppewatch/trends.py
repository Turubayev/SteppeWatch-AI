"""Тренды растительности и отделение деградации от засухи.

* Манн–Кендалл + наклон Сена — непараметрический тест монотонного тренда (векторизован
  по пикселям: ряды короткие, ~25 лет, пикселей много).
* RESTREND (Residual Trend): регрессия годового NDVI на осадки (текущего и предыдущего
  года); тренд остатков показывает изменения, которые НЕ объясняются климатом.
* RUE (Rain Use Efficiency) — NDVI на единицу осадков; устойчивое падение RUE — признак
  деградации.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def mann_kendall(y: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    """Тест Манна–Кендалла по оси времени. y: (T, ...) → словарь массивов формы (...).

    Возвращает: s, z, p (двусторонний), slope (Сен, ед./шаг), intercept.
    Пропуски (NaN) допускаются: пары с пропуском не учитываются.
    """
    y = np.moveaxis(np.asarray(y, float), axis, 0)
    T = y.shape[0]
    i, j = np.triu_indices(T, k=1)
    d = y[j] - y[i]  # (P, ...)
    valid = np.isfinite(d)
    s = np.nansum(np.sign(d), axis=0)
    n = np.isfinite(y).sum(axis=0)
    # Дисперсия с поправкой на связки (одинаковые значения).
    var = n * (n - 1) * (2 * n + 5) / 18.0
    flat = y.reshape(T, -1)
    ties_corr = np.zeros(flat.shape[1])
    for k in range(flat.shape[1]):
        col = flat[:, k][np.isfinite(flat[:, k])]
        _, counts = np.unique(np.round(col, 10), return_counts=True)
        counts = counts[counts > 1]
        ties_corr[k] = np.sum(counts * (counts - 1) * (2 * counts + 5)) / 18.0
    var = var - ties_corr.reshape(var.shape)
    z = np.where(s > 0, (s - 1) / np.sqrt(np.maximum(var, 1e-12)),
                 np.where(s < 0, (s + 1) / np.sqrt(np.maximum(var, 1e-12)), 0.0))
    p = 2 * (1 - norm.cdf(np.abs(z)))
    dt = (j - i).reshape((-1,) + (1,) * (y.ndim - 1))
    slopes = np.where(valid, d / dt, np.nan)
    slope = np.nanmedian(slopes, axis=0)
    t = np.arange(T).reshape((-1,) + (1,) * (y.ndim - 1))
    intercept = np.nanmedian(y - slope * t, axis=0)
    return {"s": s, "z": z, "p": p, "slope": slope, "intercept": intercept, "n": n}


def _lstsq_per_pixel(X: np.ndarray, y: np.ndarray):
    """X: (T, P, k), y: (T, P) → коэффициенты (P, k), прогноз (T, P). Пропуски исключаются."""
    T, P, k = X.shape
    coef = np.full((P, k), np.nan)
    fit = np.full((T, P), np.nan)
    for p in range(P):
        ok = np.isfinite(y[:, p]) & np.isfinite(X[:, p]).all(axis=1)
        if ok.sum() < k + 3:
            continue
        c, *_ = np.linalg.lstsq(X[ok, p], y[ok, p], rcond=None)
        coef[p] = c
        fit[:, p] = X[:, p] @ c
    return coef, fit


def restrend(ndvi: np.ndarray, precip: np.ndarray, lag: bool = True, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """RESTREND. ndvi: (T, H, W) годовой показатель (например, средний NDVI вегетационного сезона),
    precip: (T, H, W) или (T,) годовые осадки (мм). Возвращает карты наклона остатков, p-value,
    R² связи с осадками и класс:
      −2 — деградация, не объясняемая осадками (значимое падение остатков);
      −1 — снижение NDVI, объясняемое засухой (тренд NDVI отрицательный, остатки — нет);
       0 — стабильно;  +1 — улучшение;  NaN — нет данных.
    """
    T = ndvi.shape[0]
    shape = ndvi.shape[1:]
    y = ndvi.reshape(T, -1)
    P = y.shape[1]
    pr = np.broadcast_to(precip.reshape(T, -1) if precip.ndim > 1 else precip[:, None], (T, P))
    # Отклик растительности на осадки насыщается, поэтому регрессия — на логарифм осадков.
    pr = np.log(np.clip(pr, 1.0, None))
    cols = [np.ones((T, P)), pr]
    if lag:
        prev = np.vstack([pr[:1], pr[:-1]])  # осадки предыдущего года (запасы влаги, ветошь)
        cols.append(prev)
    X = np.stack(cols, axis=-1)
    coef, fit = _lstsq_per_pixel(X, y)
    resid = y - fit
    ss_res = np.nansum(resid**2, axis=0)
    ss_tot = np.nansum((y - np.nanmean(y, axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    mk_res = mann_kendall(resid)
    mk_raw = mann_kendall(y)
    cls = np.zeros(P)
    sig_res = mk_res["p"] < alpha
    cls[sig_res & (mk_res["slope"] < 0)] = -2
    cls[sig_res & (mk_res["slope"] > 0)] = 1
    cls[~sig_res & (mk_raw["p"] < alpha) & (mk_raw["slope"] < 0)] = -1
    cls[~np.isfinite(mk_res["slope"])] = np.nan
    return {
        "resid_slope": mk_res["slope"].reshape(shape), "resid_p": mk_res["p"].reshape(shape),
        "ndvi_slope": mk_raw["slope"].reshape(shape), "ndvi_p": mk_raw["p"].reshape(shape),
        "r2_precip": r2.reshape(shape), "class": cls.reshape(shape), "residuals": resid.reshape((T,) + shape),
        "precip_coef": coef[:, 1].reshape(shape),
    }


def rue(ndvi: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """Эффективность использования осадков: NDVI / осадки (на 100 мм)."""
    p = precip if precip.ndim == ndvi.ndim else precip.reshape((-1,) + (1,) * (ndvi.ndim - 1))
    return ndvi / np.where(p > 0, p, np.nan) * 100


CLASS_LABELS = {-2: "деградация (не объясняется осадками)", -1: "снижение из-за засухи", 0: "стабильно", 1: "улучшение"}
