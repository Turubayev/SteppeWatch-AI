"""Степные пожары: активные очаги NASA FIRMS (MODIS / VIIRS).

Нужен бесплатный ключ MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
(передаётся через переменную окружения FIRMS_MAP_KEY).
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import requests

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")


def hotspots(bbox: tuple[float, float, float, float], days: int = 3, source: str = "VIIRS_SNPP_NRT",
             date: str | None = None, map_key: str | None = None) -> pd.DataFrame:
    """Очаги в bbox (запад, юг, восток, север) за последние `days` суток (1–10) или с даты `date`."""
    key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("Не задан FIRMS_MAP_KEY (бесплатно: firms.modaps.eosdis.nasa.gov/api/map_key)")
    area = ",".join(f"{v:.4f}" for v in bbox)
    url = f"{API}/{key}/{source}/{area}/{min(max(days, 1), 10)}" + (f"/{date}" if date else "")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    if r.text.lower().startswith("invalid"):
        raise RuntimeError(r.text[:200])
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4), format="%Y-%m-%d %H%M")
    return df


def cluster_fires(df: pd.DataFrame, radius_km: float = 2.0) -> pd.DataFrame:
    """Объединение соседних пикселей-очагов в пожары (простая кластеризация по расстоянию)."""
    if df.empty:
        return df.assign(fire_id=[])
    lat, lon = np.radians(df["latitude"].to_numpy()), np.radians(df["longitude"].to_numpy())
    n = len(df)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = 6371 * np.arccos(np.clip(np.sin(lat[i]) * np.sin(lat) + np.cos(lat[i]) * np.cos(lat) * np.cos(lon - lon[i]), -1, 1))
        for j in np.flatnonzero((d < radius_km) & (np.arange(n) > i)):
            parent[find(j)] = find(i)
    out = df.copy()
    out["fire_id"] = [find(i) for i in range(n)]
    return out


def fire_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    c = cluster_fires(df)
    return c.groupby("fire_id").agg(
        first_seen=("datetime", "min"), last_seen=("datetime", "max"), hotspots=("latitude", "size"),
        lat=("latitude", "mean"), lon=("longitude", "mean"), max_frp_mw=("frp", "max"),
    ).sort_values("hotspots", ascending=False).reset_index(drop=True)
