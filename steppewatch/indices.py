"""Спектральные индексы: растительность, засоление, гари.

Каналы — отражение 0..1 (Sentinel-2 L2A: B02 синий, B03 зелёный, B04 красный, B08 БИК,
B11 SWIR1, B12 SWIR2; Landsat 8/9: SR_B2…SR_B7 после масштабирования).
"""
from __future__ import annotations

import numpy as np


def _nd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = a + b
    return np.where(s != 0, (a - b) / np.where(s != 0, s, 1), np.nan)


def ndvi(red, nir):
    return _nd(nir, red)


def nbr(nir, swir2):
    """Normalized Burn Ratio."""
    return _nd(nir, swir2)


def salinity_indices(blue, green, red, nir, swir1, swir2) -> dict[str, np.ndarray]:
    """Набор индексов засоления (Khan et al. 2005; Allbed et al. 2014; Al-Khaier 2003).

    Солончаки — яркие в видимом и SWIR, с низким NDVI; SWIR-индекс чувствителен к гипсу и солям.
    """
    blue, green, red, nir, swir1, swir2 = (np.asarray(x, float) for x in (blue, green, red, nir, swir1, swir2))
    return {
        "SI": np.sqrt(np.clip(blue * red, 0, None)),
        "SI1": np.sqrt(np.clip(green * red, 0, None)),
        "NDSI_salinity": _nd(red, nir),
        "SI_swir": _nd(swir1, swir2),
        "brightness": np.sqrt((blue**2 + green**2 + red**2 + nir**2) / 4),
    }


def salinity_class(idx: dict[str, np.ndarray], ndvi_arr: np.ndarray) -> np.ndarray:
    """Эвристическая классификация засоления: 0 — нет, 1 — слабое/умеренное, 2 — сильное (солончак).

    Пороги — стартовые; калибруются по полевым пробам (электропроводность почвы, EC).
    """
    si, bright, nds = idx["SI"], idx["brightness"], idx["NDSI_salinity"]
    out = np.zeros(np.shape(si), np.int8)
    low_veg = np.asarray(ndvi_arr) < 0.2
    out[low_veg & (si > 0.18) & (nds > -0.15)] = 1
    out[low_veg & (si > 0.28) & (bright > 0.3)] = 2
    return out


def dnbr_severity(nbr_pre, nbr_post) -> tuple[np.ndarray, np.ndarray]:
    """dNBR и классы тяжести выгорания (USGS, Key & Benson 2006):
    0 — не выгорело (<0.1), 1 — слабо (0.1–0.27), 2 — умеренно-слабо (0.27–0.44),
    3 — умеренно-сильно (0.44–0.66), 4 — сильно (>0.66)."""
    d = np.asarray(nbr_pre, float) - np.asarray(nbr_post, float)
    cls = np.digitize(d, [0.1, 0.27, 0.44, 0.66]).astype(np.int8)
    cls[~np.isfinite(d)] = -1
    return d, cls
