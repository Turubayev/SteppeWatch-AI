"""Карты засоления (Sentinel-2) и гарей (dNBR) для района.

    python scripts/salinity_burn.py --aoi data/aoi/kyzylorda.geojson --period 2024-05-01 2024-09-30   # GEE
    python scripts/salinity_burn.py --synthetic

Гари: --burn-pre 2023-05-01 2023-06-15 --burn-post 2023-08-01 2023-09-15 (композиты до и после сезона пожаров).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from steppewatch.indices import dnbr_severity, nbr, ndvi, salinity_class, salinity_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi")
    ap.add_argument("--period", nargs=2, default=["2024-05-01", "2024-09-30"])
    ap.add_argument("--burn-pre", nargs=2)
    ap.add_argument("--burn-post", nargs=2)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--name")
    a = ap.parse_args()
    pix_km2 = 0.02**2
    if a.synthetic:
        from steppewatch.synthetic import make_cube

        c = make_cube()
        b = c["bands"]
        nbr_pre, nbr_post = c["nbr_pre"], c["nbr_post"]
        name, pix_km2 = a.name or "demo", c["pixel_km"] ** 2
    else:
        from steppewatch import gee
        from steppewatch.pipeline import load_aoi

        gj, _, _ = load_aoi(a.aoi)
        name = a.name or Path(a.aoi).stem
        b = gee.sentinel2_composite(gj, *a.period)
        nbr_pre = nbr_post = None
        if a.burn_pre and a.burn_post:
            pre, post = gee.sentinel2_composite(gj, *a.burn_pre), gee.sentinel2_composite(gj, *a.burn_post)
            nbr_pre, nbr_post = nbr(pre["nir"], pre["swir2"]), nbr(post["nir"], post["swir2"])
    idx = salinity_indices(b["blue"], b["green"], b["red"], b["nir"], b["swir1"], b["swir2"])
    nd = ndvi(b["red"], b["nir"])
    sal = salinity_class(idx, nd)
    out = {"salinity_km2": {"слабое/умеренное": round(float((sal == 1).sum() * pix_km2), 1),
                            "сильное (солончак)": round(float((sal == 2).sum() * pix_km2), 1)}}
    maps = {"salinity_class": sal, "SI": idx["SI"], "ndvi": nd}
    if nbr_pre is not None:
        d, sev = dnbr_severity(nbr_pre, nbr_post)
        maps["dnbr"], maps["burn_severity"] = d, sev
        out["burned_km2"] = {lbl: round(float((sev == k).sum() * pix_km2), 1)
                             for k, lbl in [(1, "слабо"), (2, "умеренно-слабо"), (3, "умеренно-сильно"), (4, "сильно")]}
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    np.savez_compressed(res / f"{name}_salinity_burn.npz", **{k: np.asarray(v, np.float32) for k, v in maps.items()})
    (res / f"{name}_salinity_burn.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
