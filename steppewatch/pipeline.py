"""Сквозной анализ района: RESTREND → тренды → переломы → ЦУР 15.3.1 → сводка."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .breaks import break_map
from .sdg import land_cover_change, productivity, sdg_15_3_1, soc_change
from .trends import CLASS_LABELS, restrend, rue


def load_aoi(path: str | Path) -> tuple[dict, tuple[float, float, float, float], str]:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    feat = gj["features"][0] if gj.get("type") == "FeatureCollection" else gj
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
        else:
            for cc in c:
                walk(cc)

    walk(feat["geometry"]["coordinates"])
    name = feat.get("properties", {}).get("name", Path(path).stem)
    return gj, (min(xs), min(ys), max(xs), max(ys)), name


def analyze(years: np.ndarray, ndvi: np.ndarray, precip: np.ndarray, pixel_km2: float,
            baseline_years: int = 15, lc_start=None, lc_end=None, soc_start=None, soc_end=None) -> dict:
    res = restrend(ndvi, precip)
    brk = break_map(res["residuals"], alpha=0.01, min_seg=4)
    # Переломы интересны там, где есть значимое изменение; в стабильных пикселях это шум.
    brk["break_index"] = np.where(np.isin(res["class"], (-2, 1)), brk["break_index"], np.nan)
    prod = productivity(ndvi, baseline_years)
    lc = land_cover_change(lc_start, lc_end) if lc_start is not None else np.zeros(ndvi.shape[1:], int)
    soc = soc_change(soc_start, soc_end) if soc_start is not None else np.zeros(ndvi.shape[1:], int)
    sdg = sdg_15_3_1(prod["lpd"], lc, soc, pixel_km2, mask=np.isfinite(ndvi).all(axis=0))
    cls = res["class"]
    valid = np.isfinite(cls)
    areas = {CLASS_LABELS[k]: round(float((cls == k).sum() * pixel_km2), 1) for k in CLASS_LABELS}
    share = {k: round(100 * v / (valid.sum() * pixel_km2), 1) for k, v in areas.items()}
    by_year = {int(y): float(np.nanmean(ndvi[i])) for i, y in enumerate(years)}
    brk_years = brk["break_index"][np.isfinite(brk["break_index"])].astype(int)
    summary = {
        "years": [int(years[0]), int(years[-1])],
        "area_km2": round(float(valid.sum() * pixel_km2), 1),
        "restrend_classes_km2": areas,
        "restrend_classes_percent": share,
        "median_r2_ndvi_vs_precip": round(float(np.nanmedian(res["r2_precip"])), 2),
        "sdg_15_3_1": {k: v for k, v in sdg.items() if k != "degraded_mask"},
        "break_years_top": {int(years[b]): int(c) for b, c in zip(*np.unique(brk_years, return_counts=True))} if len(brk_years) else {},
        "mean_ndvi_by_year": by_year,
        "mean_precip_by_year": {int(y): float(np.nanmean(precip[i])) for i, y in enumerate(years)},
    }
    maps = {"class": cls, "resid_slope": res["resid_slope"], "resid_p": res["resid_p"], "ndvi_slope": res["ndvi_slope"],
            "r2_precip": res["r2_precip"], "break_index": brk["break_index"], "lpd": prod["lpd"],
            "sdg_degraded": sdg["degraded_mask"].astype(np.int8), "ndvi_mean": np.nanmean(ndvi, axis=0),
            "rue_mean": np.nanmean(rue(ndvi, precip), axis=0)}
    return {"summary": summary, "maps": maps}


def save(result: dict, out_dir: str | Path, name: str, years, ndvi, precip, extent=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", years=years, ndvi=ndvi.astype(np.float32),
                        precip=np.asarray(precip, np.float32), extent=np.array(extent if extent else [0, 1, 0, 1], float),
                        **{f"map_{k}": np.asarray(v, np.float32) for k, v in result["maps"].items()})
    (out / f"{name}.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return out / f"{name}.npz"


def plot(result: dict, path: str | Path, title: str, extent=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    m = result["maps"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = ax[0].imshow(m["ndvi_slope"] * 10, cmap="RdYlGn", vmin=-0.05, vmax=0.05, extent=extent)
    ax[0].set_title("Тренд NDVI, ед./10 лет (без поправки)")
    fig.colorbar(im0, ax=ax[0], shrink=0.75)
    im1 = ax[1].imshow(m["resid_slope"] * 10, cmap="RdYlGn", vmin=-0.05, vmax=0.05, extent=extent)
    ax[1].set_title("Тренд остатков RESTREND (с поправкой на осадки)")
    fig.colorbar(im1, ax=ax[1], shrink=0.75)
    cmap = ListedColormap(["#b42318", "#fdb022", "#e4e7ec", "#12b76a"])
    ax[2].imshow(m["class"], cmap=cmap, vmin=-2.5, vmax=1.5, extent=extent, interpolation="nearest")
    ax[2].set_title("Класс")
    from matplotlib.patches import Patch

    ax[2].legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, CLASS_LABELS.values())],
                 loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)
