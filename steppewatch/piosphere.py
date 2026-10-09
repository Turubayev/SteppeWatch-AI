"""Перевыпас вокруг водопоев и зимовок (пиосфера).

Скот концентрируется у воды, поэтому растительность выбивается кольцом вокруг колодцев:
NDVI растёт с расстоянием и выходит на «фоновое» плато. Аппроксимируем
    NDVI(d) = a − b · exp(−d / L)
Глубина «воронки» b и радиус L описывают нагрузку; зона перегрузки — где NDVI ниже 90 %
фонового уровня. Сравнение b и L по годам показывает, растёт ли нагрузка.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def distance_to_points(shape, transform_rc_to_km, points_rc: list[tuple[float, float]]) -> np.ndarray:
    """Расстояние (км) от каждой ячейки до ближайшей точки. transform_rc_to_km — размер ячейки, км."""
    rr, cc = np.indices(shape)
    d = np.full(shape, np.inf)
    for r, c in points_rc:
        d = np.minimum(d, np.hypot(rr - r, cc - c) * transform_rc_to_km)
    return d


def _model(d, a, b, L):
    return a - b * np.exp(-d / L)


def fit_piosphere(ndvi: np.ndarray, dist_km: np.ndarray, max_km: float = 10.0, bin_km: float = 0.25) -> dict:
    ok = np.isfinite(ndvi) & (dist_km <= max_km)
    bins = np.arange(0, max_km + bin_km, bin_km)
    idx = np.digitize(dist_km[ok], bins)
    centers, means = [], []
    for k in np.unique(idx):
        v = ndvi[ok][idx == k]
        if len(v) >= 5:
            centers.append(bins[k - 1] + bin_km / 2)
            means.append(np.median(v))
    centers, means = np.array(centers), np.array(means)
    try:
        (a, b, L), _ = curve_fit(_model, centers, means, p0=(means.max(), means.max() - means.min(), 1.0),
                                 bounds=([0, 0, 0.05], [1, 1, max_km]), maxfev=10000)
    except (RuntimeError, ValueError):
        return {"ok": False, "profile_km": centers.tolist(), "profile_ndvi": means.tolist()}
    radius = float(-L * np.log(0.1 * a / b)) if b > 0.1 * a else 0.0  # NDVI = 0.9a
    return {"ok": True, "background_ndvi": float(a), "depletion": float(b), "length_km": float(L),
            "overgrazing_radius_km": max(0.0, radius), "profile_km": centers.tolist(), "profile_ndvi": means.tolist()}


def overgrazing_mask(ndvi: np.ndarray, dist_km: np.ndarray, fit: dict) -> np.ndarray:
    if not fit.get("ok"):
        return np.zeros(ndvi.shape, bool)
    return (dist_km <= fit["overgrazing_radius_km"]) & (ndvi < 0.9 * fit["background_ndvi"])
