"""Индикатор ЦУР 15.3.1 «Доля деградированных земель» по методике UNCCD (Good Practice Guidance v2).

Три субиндикатора, принцип «один против всех» (one-out-all-out): пиксель деградирован,
если деградирован хотя бы по одному из них.

1. Динамика продуктивности (LPD) — из трёх метрик NDVI:
   * trajectory: значимый тренд (Манн–Кендалл) за отчётный период;
   * state: среднее за последние 3 года против распределения базового периода (z < −1.96 / > 1.96);
   * performance: продуктивность пикселя против 90-го перцентиля в той же единице «почва × покров»
     (< 0.5 — деградация).
   Комбинация по упрощённой таблице GPG: деградация, если trajectory < 0, или state < 0 и
   performance низкая.
2. Изменение почвенно-растительного покрова — матрица переходов по 7 классам IPCC.
3. Запас органического углерода почвы (SOC) — потеря ≥ 10 % за период = деградация.
"""
from __future__ import annotations

import numpy as np

from .trends import mann_kendall

# 7 классов IPCC: 1 лес, 2 луга/степь, 3 пашня, 4 водно-болотные, 5 застройка, 6 прочие (голые), 7 вода
IPCC = {1: "Лес", 2: "Травянистые (степь)", 3: "Пашня", 4: "Водно-болотные", 5: "Застройка", 6: "Прочие земли", 7: "Вода"}
# Матрица UNCCD по умолчанию: строка — было, столбец — стало; −1 деградация, 0 стабильно, +1 улучшение.
# Значения можно заменить национальной матрицей (как в Trends.Earth: «Custom transition matrix»).
LC_MATRIX = np.array([
    [0, -1, -1, -1, -1, -1, 0],
    [1, 0, 1, -1, -1, -1, 0],
    [1, -1, 0, -1, -1, -1, 0],
    [-1, -1, -1, 0, -1, -1, 0],
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

# ESA CCI / WorldCover → IPCC (основные классы, остальные по таблице Trends.Earth)
WORLDCOVER_TO_IPCC = {10: 1, 20: 2, 30: 2, 40: 3, 50: 5, 60: 6, 70: 6, 80: 7, 90: 4, 95: 4, 100: 2}


def productivity(ndvi: np.ndarray, baseline_years: int, alpha: float = 0.05, units: np.ndarray | None = None) -> dict:
    """ndvi: (T, H, W) годовые значения. Первые baseline_years — базовый период."""
    ndvi = np.asarray(ndvi, float)
    if ndvi.ndim < 2 or not 3 <= baseline_years <= len(ndvi) - 3 or not 0 < alpha < 1:
        raise ValueError('Need >=3 baseline years and >=3 subsequent years; alpha in (0,1)')
    mk = mann_kendall(ndvi)
    trajectory = np.where(mk["p"] < alpha, np.sign(mk["slope"]), 0)
    base = ndvi[:baseline_years]
    recent = np.nanmean(ndvi[-3:], axis=0)
    z = (recent - np.nanmean(base, axis=0)) / np.where(np.nanstd(base, axis=0) > 0, np.nanstd(base, axis=0), np.nan)
    state = np.where(z < -1.96, -1, np.where(z > 1.96, 1, 0))
    mean_all = np.nanmean(ndvi, axis=0)
    units = np.zeros(mean_all.shape, int) if units is None else units
    perf = np.zeros(mean_all.shape)
    for u in np.unique(units):
        m = units == u
        p90 = np.nanpercentile(mean_all[m], 90) if np.isfinite(mean_all[m]).any() else np.nan
        perf[m] = mean_all[m] / p90
    performance = np.where(perf < 0.5, -1, 0)
    lpd = np.zeros(mean_all.shape, np.int8)
    lpd[(trajectory > 0) | (state > 0)] = 1
    lpd[(trajectory < 0) | ((state < 0) & (performance < 0))] = -1
    lpd[~np.isfinite(ndvi).all(axis=0)] = -9
    return {"trajectory": trajectory, "state": state, "performance": performance, "lpd": lpd, "z_state": z}


def land_cover_change(lc_start: np.ndarray, lc_end: np.ndarray) -> np.ndarray:
    """Классы IPCC (1–7) на начало и конец → −1/0/+1 по матрице UNCCD."""
    a, b = np.broadcast_arrays(np.asarray(lc_start, float), np.asarray(lc_end, float))
    valid = np.isin(a, np.arange(1,8)) & np.isin(b, np.arange(1,8))
    out = np.full(a.shape, -9, dtype=np.int8)
    out[valid] = LC_MATRIX[a[valid].astype(int)-1, b[valid].astype(int)-1]
    return out


def soc_change(soc_start: np.ndarray, soc_end: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    if not 0 < threshold < 1: raise ValueError('SOC threshold must be in (0,1)')
    start, end = np.broadcast_arrays(np.asarray(soc_start, float), np.asarray(soc_end, float))
    valid = np.isfinite(start) & np.isfinite(end) & (start > 0) & (end >= 0)
    rel = (end - start) / np.where(valid, start, np.nan)
    return np.where(valid, np.where(rel <= -threshold, -1, np.where(rel >= threshold, 1, 0)), -9)


def sdg_15_3_1(lpd: np.ndarray, lc: np.ndarray, soc: np.ndarray, pixel_area_km2: float | np.ndarray = 1.0,
               mask: np.ndarray | None = None) -> dict:
    """Итог: доля деградированных земель (вода исключается маской)."""
    lpd = np.asarray(lpd)
    lc = np.full(lpd.shape, -9) if lc is None else np.asarray(lc)
    soc = np.full(lpd.shape, -9) if soc is None else np.asarray(soc)
    if lc.shape != lpd.shape or soc.shape != lpd.shape: raise ValueError('Subindicator grids must match')
    for a in (lpd,lc,soc):
        if not np.isin(a,[-9,-1,0,1]).all(): raise ValueError('Subindicators must be -9,-1,0,1')
    candidate = (lpd != -9) if mask is None else (np.asarray(mask,bool) & (lpd != -9))
    valid = candidate & (lc != -9) & (soc != -9)
    degraded = ((lpd == -1) | (lc == -1) | (soc == -1)) & valid
    improved = ((lpd == 1) | (lc == 1) | (soc == 1)) & ~degraded & valid
    area = np.broadcast_to(pixel_area_km2, lpd.shape)
    if not (np.isfinite(area) & (area > 0)).all(): raise ValueError('Pixel area must be finite and positive')
    total = float(area[valid].sum())
    candidate_area = float(area[candidate].sum())
    complete = bool(candidate.any() and (valid[candidate]).all())
    screening = ((lpd == -1) | (lc == -1) | (soc == -1)) & candidate
    return {
        "degraded_km2": round(float(area[degraded].sum()), 1),
        "improved_km2": round(float(area[improved].sum()), 1),
        "total_km2": round(total, 1),
        "sdg_15_3_1_percent": round(100 * float(area[degraded].sum()) / total, 2) if total and complete else None,
        "status": "complete_subindicators_simplified_method" if complete else "incomplete_subindicators",
        "coverage_percent": round(100 * total / candidate_area, 2) if candidate_area else 0.0,
        "screening_degraded_percent": round(100 * float(area[screening].sum()) / candidate_area, 2) if candidate_area else None,
        "note": "Screening is a lower bound from available subindicators; this simplified method is not a certified UNCCD report.",
        "by_subindicator_percent": {
            "productivity": round(100 * float(area[(lpd == -1) & valid].sum()) / total, 2) if total else None,
            "land_cover": round(100 * float(area[(lc == -1) & valid].sum()) / total, 2) if total else None,
            "soc": round(100 * float(area[(soc == -1) & valid].sum()) / total, 2) if total else None,
        },
        "degraded_mask": degraded,
    }
