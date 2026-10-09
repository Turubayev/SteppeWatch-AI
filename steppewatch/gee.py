"""Google Earth Engine: годовые ряды NDVI (MODIS MOD13Q1) и осадков (CHIRPS) по полигону.

Требуется аккаунт GEE и зарегистрированный Cloud-проект (некоммерческое использование бесплатно):
    earthengine authenticate
    export EE_PROJECT=my-cloud-project

Все тяжёлые вычисления (композиты, суммы осадков) выполняются на стороне GEE; в Python
приходят только готовые сетки умеренного размера (sampleRectangle, до ~262 тыс. пикселей).
"""
from __future__ import annotations

import os

import numpy as np


def init(project: str | None = None):
    import ee

    ee.Initialize(project=project or os.environ.get("EE_PROJECT"))
    return ee


def _region(ee, geojson: dict):
    geom = geojson["features"][0]["geometry"] if geojson.get("type") == "FeatureCollection" else geojson.get("geometry", geojson)
    return ee.Geometry(geom)


def annual_stack(geojson: dict, start: int, end: int, season=(4, 9), scale: int = 250, project: str | None = None) -> dict:
    """Средний NDVI вегетационного сезона и годовые осадки по годам → numpy (T, H, W)."""
    ee = init(project)
    region = _region(ee, geojson)
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")

    def year_img(y):
        y = ee.Number(y)
        nd = (modis.filter(ee.Filter.calendarRange(y, y, "year"))
              .filter(ee.Filter.calendarRange(season[0], season[1], "month")).mean().multiply(0.0001).rename("ndvi"))
        # Гидрологический год: октябрь прошлого — сентябрь текущего.
        pr = chirps.filterDate(ee.Date.fromYMD(y.subtract(1), 10, 1), ee.Date.fromYMD(y, 10, 1)).sum().rename("precip")
        return nd.addBands(pr.resample("bilinear").reproject(nd.projection().atScale(scale)))

    ndvi, precip = [], []
    for y in range(start, end + 1):
        arr = year_img(y).sampleRectangle(region=region, defaultValue=-9999).getInfo()["properties"]
        ndvi.append(np.array(arr["ndvi"], float))
        precip.append(np.array(arr["precip"], float))
    nd = np.where(np.stack(ndvi) == -9999, np.nan, np.stack(ndvi))
    pr = np.where(np.stack(precip) == -9999, np.nan, np.stack(precip))
    return {"years": np.arange(start, end + 1), "ndvi": nd, "precip": pr, "source": "GEE: MOD13Q1 + CHIRPS"}


def sentinel2_composite(geojson: dict, start: str, end: str, scale: int = 20, project: str | None = None) -> dict:
    """Безоблачный медианный композит Sentinel-2 L2A (каналы для индексов засоления)."""
    ee = init(project)
    region = _region(ee, geojson)

    def mask(img):
        scl = img.select("SCL")
        return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))).divide(10000)

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask))
    img = col.select(["B2", "B3", "B4", "B8", "B11", "B12"], ["blue", "green", "red", "nir", "swir1", "swir2"]).median()
    arr = img.reproject("EPSG:4326", None, scale).sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
    return {k: np.array(v, float) for k, v in arr.items()}


def burned_area(geojson: dict, year: int, project: str | None = None) -> dict:
    """Площадь гарей MCD64A1 (500 м) за год, км²."""
    ee = init(project)
    region = _region(ee, geojson)
    ba = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(f"{year}-01-01", f"{year + 1}-01-01").select("BurnDate").max()
    area = ba.gt(0).multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), region, 500, maxPixels=1e10)
    return {"year": year, "burned_km2": (area.getInfo().get("BurnDate") or 0) / 1e6}


def land_cover_soc(geojson: dict, project: str | None = None) -> dict:
    """ESA WorldCover 2020/2021 и SOC SoilGrids (0–30 см) для субиндикаторов ЦУР 15.3.1."""
    ee = init(project)
    region = _region(ee, geojson)
    wc20 = ee.ImageCollection("ESA/WorldCover/v100").first().rename("lc2020")
    wc21 = ee.ImageCollection("ESA/WorldCover/v200").first().rename("lc2021")
    soc = ee.Image("projects/soilgrids-isric/ocs_mean").select("ocs_0-30cm_mean").rename("soc")
    img = wc20.addBands(wc21).addBands(soc).reproject("EPSG:4326", None, 250)
    arr = img.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
    return {k: np.array(v) for k, v in arr.items()}
