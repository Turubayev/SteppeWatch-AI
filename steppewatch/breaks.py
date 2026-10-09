"""Точки перелома (упрощённый BFAST для годовых рядов).

Полный BFAST (Verbesselt et al., 2010) раскладывает 16-дневный ряд на тренд, сезонность
и остаток и ищет несколько переломов. Для годовых рядов остатков RESTREND (где сезонность
уже снята агрегированием) достаточно кусочно-линейной модели с одним переломом:
выбираем год, минимизирующий сумму квадратов ошибок, и проверяем значимость F-тестом
Чоу против модели без перелома. Для 16-дневных рядов есть вариант с гармониками.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import f as f_dist


def _rss(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    return float(r @ r), c


def single_break(y: np.ndarray, min_seg: int = 4, harmonics: int = 0, period: float = 23.0) -> dict:
    """Один перелом в ряду y (1D). Возвращает индекс, p-value, сдвиг уровня и изменение наклона."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    t = np.arange(len(y), dtype=float)
    t, y = t[ok], y[ok]
    n = len(y)
    if n < 2 * min_seg + 2:
        return {"break": None, "p": np.nan}

    def design(tt, b=None):
        cols = [np.ones_like(tt), tt]
        for h in range(1, harmonics + 1):
            cols += [np.sin(2 * np.pi * h * tt / period), np.cos(2 * np.pi * h * tt / period)]
        if b is not None:
            step = (tt >= b).astype(float)
            cols += [step, step * (tt - b)]
        return np.stack(cols, axis=1)

    rss0, _ = _rss(design(t), y)
    best = (np.inf, None, None)
    for k in range(min_seg, n - min_seg):
        rss1, c = _rss(design(t, t[k]), y)
        if rss1 < best[0]:
            best = (rss1, t[k], c)
    rss1, b, c = best
    q = 2  # добавленные параметры
    k_full = design(t, b).shape[1]
    F = ((rss0 - rss1) / q) / (rss1 / max(n - k_full, 1))
    p = 1 - f_dist.cdf(F, q, max(n - k_full, 1))
    return {"break": int(b), "p": float(p), "level_shift": float(c[-2]), "slope_change": float(c[-1]),
            "slope_before": float(c[1]), "slope_after": float(c[1] + c[-1])}


def break_map(series: np.ndarray, alpha: float = 0.05, **kw) -> dict[str, np.ndarray]:
    """Перелом для каждого пикселя: series (T, H, W) → карта индекса перелома (NaN, если незначим)."""
    T = series.shape[0]
    flat = series.reshape(T, -1)
    idx = np.full(flat.shape[1], np.nan)
    shift = np.full(flat.shape[1], np.nan)
    for p in range(flat.shape[1]):
        r = single_break(flat[:, p], **kw)
        if r["break"] is not None and r["p"] < alpha:
            idx[p] = r["break"]
            shift[p] = r["slope_after"] - r["slope_before"] + r["level_shift"]
    return {"break_index": idx.reshape(series.shape[1:]), "change": shift.reshape(series.shape[1:])}
