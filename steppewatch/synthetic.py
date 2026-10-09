"""Синтетический «кубик» данных с заранее известной правдой — для тестов и демо без GEE.

Сцена: степь 120×120 пикселей (MODIS 250 м ≈ 30×30 км), 2000–2024 гг.
* межгодовая изменчивость осадков с засухами (2012, 2021 — как в реальной степи);
* NDVI = f(осадки) + шум; поверх — известные процессы:
  - перевыпас: «воронка» вокруг двух колодцев, усиливающаяся после 2008 г.;
  - засоление: пятно на юго-западе с падением NDVI и ростом яркости/SI после 2010 г.;
  - пожар 2017 г.: резкий провал NDVI с восстановлением, выгорание (dNBR);
  - восстановление залежи на северо-востоке.
Возвращает словарь массивов и маску «правды» для проверки RESTREND и классификатора.
"""
from __future__ import annotations

import numpy as np


def make_cube(n: int = 120, years: range = range(2000, 2025), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    yrs = np.array(list(years))
    T = len(yrs)
    rr, cc = np.indices((n, n))
    # Осадки: градиент с юга на север (север влажнее), годовая изменчивость, засухи.
    year_factor = rng.normal(1.0, 0.18, T)
    for dry in (2012, 2021):
        if dry in yrs:
            year_factor[yrs == dry] = 0.55
    base_p = 180 + 120 * (1 - rr / n)  # мм/год
    precip = base_p[None] * year_factor[:, None, None] * rng.normal(1, 0.04, (T, n, n))
    # Аридизация на востоке: осадки падают на 35 % за период — NDVI снижается, но это климат,
    # а не деградация (RESTREND должен отнести это к «снижению из-за засухи»).
    drying = (cc > 105) & (rr < 90)
    precip *= 1 - 0.35 * drying[None] * np.linspace(0, 1, T)[:, None, None]
    # Растительность реагирует на осадки (насыщение), шум.
    potential = 0.08 + 0.45 * (1 - np.exp(-precip / 260))
    ndvi = potential + rng.normal(0, 0.012, (T, n, n))

    truth = np.zeros((n, n), np.int8)  # 0 нет, 1 засоление, 2 перевыпас, 3 эрозия, 4 пожар, 5 улучшение, 6 засуха
    wells = [(35, 40), (80, 85)]
    villages = [(30, 95)]
    dist_w = np.min([np.hypot(rr - r, cc - c) for r, c in wells], axis=0) * 0.25  # км
    t_rel = np.clip(yrs - 2008, 0, None)[:, None, None]
    graze = 0.18 * np.exp(-dist_w / 1.5)[None] * np.minimum(t_rel / 10, 1)
    ndvi -= graze
    truth[(dist_w < 3.7)] = 2  # амплитуда выбивания > 0.015 NDVI
    salt = (np.hypot(rr - 95, cc - 25) < 12)
    salt_t = np.clip(yrs - 2010, 0, None)[:, None, None] / 14
    ndvi -= 0.14 * salt[None] * salt_t
    truth[salt] = 1
    eros = (rr < 18) & (cc > 40) & (cc < 75)  # овраги/дефляция на склоне
    ndvi -= 0.08 * eros[None] * np.linspace(0, 1, T)[:, None, None]
    truth[eros] = 3
    burn = (np.hypot(rr - 60, cc - 55) < 14)
    if 2017 in yrs:
        k = int(np.flatnonzero(yrs == 2017)[0])
        rec = np.exp(-np.arange(T - k) / 2.0)
        ndvi[k:] -= 0.2 * burn[None] * rec[:, None, None]
    truth[burn & (truth == 0)] = 4
    rest = (rr > 95) & (cc > 90)
    ndvi += 0.08 * rest[None] * np.linspace(0, 1, T)[:, None, None]
    truth[rest] = 5
    truth[drying & (truth == 0)] = 6
    ndvi = np.clip(ndvi, 0.02, 0.9)

    # Спектр последнего года (Sentinel-2-подобный) для индексов засоления.
    veg = ndvi[-1]
    soil = 0.18 + 0.05 * rng.random((n, n))
    red = soil * (1 - veg) + 0.04 * veg
    nir = red * (1 + veg) / np.clip(1 - veg, 0.05, None)
    blue, green = red * 0.75, red * 0.9
    swir1, swir2 = soil * 1.3, soil * 1.1
    blue, green, red = (x + 0.2 * salt for x in (blue, green, red))
    swir1 = swir1 + 0.12 * salt
    swir2 = swir2 + 0.02 * salt
    nbr_pre = 0.35 - 0.1 * rng.random((n, n))
    nbr_post = nbr_pre - 0.5 * burn
    return {"years": yrs, "ndvi": ndvi.astype(np.float32), "precip": precip.astype(np.float32), "truth": truth,
            "bands": {"blue": blue, "green": green, "red": red, "nir": nir, "swir1": swir1, "swir2": swir2},
            "nbr_pre": nbr_pre, "nbr_post": nbr_post, "wells_rc": wells, "villages_rc": villages,
            "pixel_km": 0.25, "lc_start": np.full((n, n), 2), "lc_end": np.where(salt, 6, 2),
            "soc_start": np.full((n, n), 30.0), "soc_end": np.where(salt | eros, 25.0, 30.0)}
