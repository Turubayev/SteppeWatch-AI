"""Классификация типа деградации: засоление / перевыпас / эрозия / пожар / нет.

Признаки пикселя: тренды (NDVI и остатки RESTREND), перелом, индексы засоления,
расстояние до водопоев и аулов, история пожаров, рельеф. Метки — полевые точки
(геоботанические обследования, институты почвоведения, экспертная разметка по снимкам).

Проверка — пространственная: модель обучается на одних районах и проверяется на другом
(`spatial_cv`), иначе соседние пиксели «подсказывают» ответ и точность завышается.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["нет деградации", "засоление", "перевыпас", "эрозия", "пожар"]
FEATURES = ["ndvi_slope", "resid_slope", "resid_p", "r2_precip", "ndvi_mean", "break_change",
            "SI", "NDSI_salinity", "SI_swir", "brightness", "dist_water_km", "dist_village_km",
            "fires_count", "dnbr_max", "slope_deg"]


def _model(seed: int = 0):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed)


class DegradationClassifier:
    def __init__(self, features: list[str] = FEATURES):
        self.features = features
        self.model = None

    def fit(self, df: pd.DataFrame, label_col: str = "label"):
        self.model = _model().fit(df[self.features], df[label_col])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path) -> "DegradationClassifier":
        return pickle.loads(Path(path).read_bytes())


def spatial_cv(df: pd.DataFrame, group_col: str = "district", label_col: str = "label") -> dict:
    """Leave-one-district-out: precision / recall / F1 по классам на каждом отложенном районе."""
    from sklearn.metrics import classification_report

    reports, y_all, p_all = {}, [], []
    for g in sorted(df[group_col].unique()):
        tr, te = df[df[group_col] != g], df[df[group_col] == g]
        clf = DegradationClassifier().fit(tr, label_col)
        p = clf.predict(te)
        y_all += list(te[label_col])
        p_all += list(p)
        reports[g] = classification_report(te[label_col], p, labels=CLASSES, output_dict=True, zero_division=0)["macro avg"]
    overall = classification_report(y_all, p_all, labels=CLASSES, output_dict=True, zero_division=0)
    return {"by_heldout_district": reports, "overall": {c: overall[c] for c in CLASSES + ["macro avg"]}}
