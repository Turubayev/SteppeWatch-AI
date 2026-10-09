"""Открытый стек без Google Earth Engine: STAC (Microsoft Planetary Computer) + Open-Meteo.

* NDVI: MODIS MOD13Q1 v6.1 (коллекция `modis-13Q1-061`, 250 м, 16 сут) — читаются только
  окна нужного района из Cloud-Optimized GeoTIFF.
* Осадки: ERA5 через Open-Meteo (средние по району; для небольших районов RESTREND
  с общим рядом осадков — допустимое упрощение, для крупных — используйте CHIRPS через GEE).
Тот же подход (STAC + xarray/rasterio) используется в DalaFlood-Cast.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def _cached(name: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{hashlib.md5(name.encode()).hexdigest()}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    out = fn()
    np.savez_compressed(p, **out)
    return out


def modis_ndvi_composite(bbox: tuple[float, float, float, float], date_range: str, max_items: int = 4) -> dict:
    """Медиана NDVI MOD13Q1 по сценам за период в окне bbox (WGS84) → массив + привязка."""
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.warp import Resampling, reproject

    def run():
        cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        items = list(cat.search(collections=["modis-13Q1-061"], bbox=bbox, datetime=date_range, max_items=max_items).items())
        if not items:
            raise RuntimeError(f"Нет сцен MOD13Q1 за {date_range}")
        w, s, e, n = bbox
        res = 0.0025  # ≈ 250 м
        width, height = int(round((e - w) / res)), int(round((n - s) / res))
        dst_transform = rasterio.transform.from_bounds(w, s, e, n, width, height)
        stack = []
        for it in items:
            with rasterio.open(it.assets["250m_16_days_NDVI"].href) as src:
                dst = np.full((height, width), np.nan, np.float32)
                reproject(rasterio.band(src, 1), dst, dst_transform=dst_transform, dst_crs="EPSG:4326",
                          resampling=Resampling.nearest, src_nodata=-3000, dst_nodata=np.nan)
                stack.append(dst * 0.0001)
        arr = np.nanmedian(np.stack(stack), axis=0)
        return {"ndvi": arr.astype(np.float32), "transform": np.array(dst_transform)[:6]}

    return _cached(f"modis{bbox}{date_range}{max_items}", run)


def annual_ndvi(bbox, start: int, end: int, months=(5, 6, 7, 8), verbose: bool = True) -> dict:
    """Средний NDVI вегетационного сезона по годам: по одному медианному композиту на месяц."""
    years, out, transform = [], [], None
    for y in range(start, end + 1):
        monthly = []
        for m in months:
            last = (np.datetime64(f"{y}-{m:02d}-01") + np.timedelta64(31, "D")).astype("datetime64[M]") - np.timedelta64(1, "D")
            c = modis_ndvi_composite(bbox, f"{y}-{m:02d}-01/{last}", max_items=4)
            monthly.append(c["ndvi"])
            transform = c["transform"]
        out.append(np.nanmean(np.stack(monthly), axis=0))
        years.append(y)
        if verbose:
            print(f"  NDVI {y}: среднее {np.nanmean(out[-1]):.3f}", flush=True)
    return {"years": np.array(years), "ndvi": np.stack(out), "transform": transform}


def annual_precip(lat: float, lon: float, start: int, end: int) -> np.ndarray:
    """Осадки гидрологического года (октябрь–сентябрь), мм — ERA5 через Open-Meteo."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"precip_{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    if key.exists():
        js = json.loads(key.read_text())
    else:
        params = {"latitude": lat, "longitude": lon, "start_date": f"{start - 1}-10-01", "end_date": f"{end}-09-30",
                  "daily": "precipitation_sum", "models": "era5", "timezone": "auto"}
        for i in range(4):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
            if r.status_code == 429 and i < 3:
                time.sleep(20)
                continue
            r.raise_for_status()
            break
        js = r.json()
        key.write_text(json.dumps(js))
    t = np.array(js["daily"]["time"], dtype="datetime64[D]")
    p = np.array([np.nan if v is None else v for v in js["daily"]["precipitation_sum"]], float)
    hy = t.astype("datetime64[Y]").astype(int) + 1970 + (t.astype("datetime64[M]").astype(int) % 12 >= 9)
    return np.array([np.nansum(p[hy == y]) for y in range(start, end + 1)])
