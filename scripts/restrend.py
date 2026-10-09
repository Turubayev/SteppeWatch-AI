"""Деградация земель с поправкой на осадки (RESTREND), тренды, переломы и ЦУР 15.3.1 для района.

    # Google Earth Engine (MOD13Q1 + CHIRPS)
    python scripts/restrend.py --aoi data/aoi/kyzylorda.geojson --start 2000 --end 2025 --source gee
    # Открытый стек: MODIS через STAC (Planetary Computer) + осадки ERA5 (Open-Meteo), без регистрации
    python scripts/restrend.py --aoi data/aoi/kyzylorda.geojson --start 2001 --end 2024 --source stac
    # Синтетика с известной «правдой» (без интернета)
    python scripts/restrend.py --source synthetic --name demo

Результаты: results/<name>.npz (карты для дашборда), <name>.json (сводка), <name>.png.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from steppewatch.pipeline import analyze, load_aoi, plot, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi")
    ap.add_argument("--start", type=int, default=2001)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--source", default="stac", choices=["gee", "stac", "synthetic"])
    ap.add_argument("--baseline", type=int, default=15, help="лет базового периода для ЦУР 15.3.1")
    ap.add_argument("--name")
    a = ap.parse_args()

    extra, extent = {}, None
    if a.source == "synthetic":
        from steppewatch.synthetic import make_cube

        c = make_cube()
        years, ndvi, precip, pix_km2 = c["years"], c["ndvi"], c["precip"], c["pixel_km"] ** 2
        extra = {k: c[k] for k in ("lc_start", "lc_end", "soc_start", "soc_end")}
        title, name = "Синтетический район (известная правда)", a.name or "demo"
    else:
        if not a.aoi:
            ap.error("укажите --aoi")
        gj, bbox, title = load_aoi(a.aoi)
        name = a.name or Path(a.aoi).stem
        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        lat_c = (bbox[1] + bbox[3]) / 2
        if a.source == "gee":
            from steppewatch import gee

            st = gee.annual_stack(gj, a.start, a.end)
            years, ndvi, precip = st["years"], st["ndvi"], st["precip"]
            pix_km2 = 0.25**2
            try:
                lc = gee.land_cover_soc(gj)
                from steppewatch.sdg import WORLDCOVER_TO_IPCC

                to_ipcc = np.vectorize(lambda v: WORLDCOVER_TO_IPCC.get(int(v), 6))
                extra = {"lc_start": to_ipcc(lc["lc2020"]), "lc_end": to_ipcc(lc["lc2021"])}
            except Exception as e:  # noqa: BLE001
                print("Покров/SOC не получены:", e)
        else:
            from steppewatch import opensource

            print(f"MODIS NDVI {a.start}–{a.end} для «{title}» (Planetary Computer)…", flush=True)
            st = opensource.annual_ndvi(bbox, a.start, a.end)
            years, ndvi = st["years"], st["ndvi"]
            pr = opensource.annual_precip(lat_c, (bbox[0] + bbox[2]) / 2, a.start, a.end)
            precip = pr  # один ряд осадков на район (ERA5); пиксельные осадки — через --source gee
            pix_km2 = (0.0025 * 111.32) * (0.0025 * 111.32 * np.cos(np.radians(lat_c)))
    res = analyze(years, ndvi, np.asarray(precip, float), pix_km2, a.baseline, **extra)
    out = ROOT / "results"
    save(res, out, name, years, ndvi, precip, extent)
    plot(res, out / f"{name}.png", f"{title}: {years[0]}–{years[-1]}", extent)
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("area_km2", "restrend_classes_percent", "median_r2_ndvi_vs_precip", "sdg_15_3_1",
                                         "break_years_top")}, ensure_ascii=False, indent=1))
    if a.source == "synthetic":
        from steppewatch.synthetic import make_cube

        truth = make_cube()["truth"]
        deg = res["maps"]["class"] == -2
        true_deg = np.isin(truth, [1, 2, 3])
        tp = (deg & true_deg).sum()
        print(f"Проверка на синтетике: precision {tp / max(deg.sum(), 1):.2f}, recall {tp / max(true_deg.sum(), 1):.2f}; "
              f"доля засушливого снижения, ошибочно названного деградацией: "
              f"{(deg & (truth == 0)).sum() / max((truth == 0).sum(), 1):.3f}")


if __name__ == "__main__":
    main()
