"""Классификатор типа деградации по полевым точкам.

Формат разметки (CSV): lat, lon (или row, col), district, label, + признаки steppewatch.classify.FEATURES.
label ∈ {нет деградации, засоление, перевыпас, эрозия, пожар}.

    python scripts/train_classifier.py --labels data/field_points.csv
    python scripts/train_classifier.py --synthetic        # демо: признаки и метки из синтетической сцены

Проверка — по отложенным районам (leave-one-district-out), результат: results/classifier_cv.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from steppewatch.classify import CLASSES, DegradationClassifier, spatial_cv  # noqa: E402


def synthetic_points(n_points: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Признаки по синтетической сцене + «полевые точки»; районы = полосы сцены."""
    from steppewatch.breaks import break_map
    from steppewatch.indices import dnbr_severity, salinity_indices
    from steppewatch.piosphere import distance_to_points
    from steppewatch.synthetic import make_cube
    from steppewatch.trends import restrend

    c = make_cube(seed=seed)
    r = restrend(c["ndvi"], c["precip"])
    b = c["bands"]
    idx = salinity_indices(b["blue"], b["green"], b["red"], b["nir"], b["swir1"], b["swir2"])
    dnbr, _ = dnbr_severity(c["nbr_pre"], c["nbr_post"])
    n = c["ndvi"].shape[1]
    feats = {
        "ndvi_slope": r["ndvi_slope"], "resid_slope": r["resid_slope"], "resid_p": r["resid_p"], "r2_precip": r["r2_precip"],
        "ndvi_mean": c["ndvi"].mean(0), "break_change": np.nan_to_num(break_map(r["residuals"], alpha=0.05)["change"]),
        **{k: idx[k] for k in ("SI", "NDSI_salinity", "SI_swir", "brightness")},
        "dist_water_km": distance_to_points((n, n), c["pixel_km"], c["wells_rc"]),
        "dist_village_km": distance_to_points((n, n), c["pixel_km"], c["villages_rc"]),
        "fires_count": (dnbr > 0.27).astype(float), "dnbr_max": dnbr,
        "slope_deg": np.where(np.indices((n, n))[0] < 18, 6.0, 1.0) + np.random.default_rng(seed).normal(0, 0.5, (n, n)),
    }
    rng = np.random.default_rng(seed + 1)
    rr, cc = rng.integers(0, n, n_points), rng.integers(0, n, n_points)
    truth = c["truth"][rr, cc]
    label_map = {0: "нет деградации", 1: "засоление", 2: "перевыпас", 3: "эрозия", 4: "пожар", 5: "нет деградации", 6: "нет деградации"}
    df = pd.DataFrame({k: v[rr, cc] for k, v in feats.items()})
    df["label"] = [label_map[int(t)] for t in truth]
    # «Районы» — вертикальные полосы сцены; каждый процесс встречается хотя бы в двух из них,
    # иначе при отложенном районе модели не на чем учиться (реальная проблема полевой разметки).
    df["district"] = [f"р-н {k + 1}" for k in cc * 5 // n]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        df = synthetic_points()
    elif a.labels:
        df = pd.read_csv(a.labels)
    else:
        ap.error("укажите --labels или --synthetic")
    print(df["label"].value_counts().to_string())
    cv = spatial_cv(df)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "classifier_cv.json").write_text(json.dumps(cv, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nПроверка на отложенных районах (F1):")
    for c in CLASSES:
        o = cv["overall"][c]
        print(f"  {c:16s} precision {o['precision']:.2f}  recall {o['recall']:.2f}  F1 {o['f1-score']:.2f}  (n={int(o['support'])})")
    clf = DegradationClassifier().fit(df)
    (ROOT / "models").mkdir(exist_ok=True)
    clf.save(ROOT / "models" / "degradation_classifier.pkl")


if __name__ == "__main__":
    main()
