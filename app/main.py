"""Дашборд SteppeWatch-AI (Streamlit).

    streamlit run app/main.py

Показывает результаты scripts/restrend.py (results/*.npz + *.json): карты трендов
с поправкой на осадки, классы деградации, ЦУР 15.3.1, ряды NDVI и осадков, переломы.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from steppewatch.trends import CLASS_LABELS  # noqa: E402

RESULTS = ROOT / "results"
st.set_page_config(page_title="SteppeWatch-AI", page_icon="🌾", layout="wide")
st.title("🌾 SteppeWatch-AI — деградация земель с поправкой на засуху")
st.caption("«Деградация» на карте — сигнал для полевой проверки, а не доказательство нарушения землепользования.")

runs = sorted(p.stem for p in RESULTS.glob("*.npz") if not p.stem.endswith("_salinity_burn"))
if not runs:
    st.warning("Нет результатов. Запустите, например: `python scripts/restrend.py --source synthetic`")
    st.stop()
run = st.sidebar.selectbox("Район / расчёт", runs)
z = np.load(RESULTS / f"{run}.npz")
summary = json.loads((RESULTS / f"{run}.json").read_text(encoding="utf-8"))
years = z["years"]
extent = z["extent"].tolist() if "extent" in z and z["extent"].tolist() != [0, 1, 0, 1] else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Площадь", f"{summary['area_km2']:,.0f} км²".replace(",", " "))
sdg_summary=summary['sdg_15_3_1']
sdg_value=sdg_summary.get('sdg_15_3_1_percent') if 'status' in sdg_summary else None
def percent(value):
    return f'{value:.1f} %' if isinstance(value,(int,float)) and np.isfinite(value) else 'Нет полного расчёта'
c2.metric('ЦУР 15.3.1 (упрощённый расчёт)', percent(sdg_value))
if sdg_value is None:
    st.info('Для полного показателя нужны покров и SOC. Старые результаты нужно пересчитать; отсутствие данных не означает стабильность.')

pct = summary["restrend_classes_percent"]
c3.metric('Снижение после поправки на осадки', percent(pct.get(CLASS_LABELS[-2], pct.get('деградация (не объясняется осадками)'))))
c4.metric('Снижение без значимого остаточного тренда', percent(pct.get(CLASS_LABELS[-1], pct.get('снижение из-за засухи'))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # Streamlit выполняет скрипт в отдельном потоке: GUI-бэкенды там падают
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

layer = st.sidebar.radio("Слой карты", ["Класс RESTREND", "Тренд остатков", "Тренд NDVI (без поправки)",
                                        "Связь NDVI с осадками (R²)", "Год перелома", "ЦУР 15.3.1", "Средний NDVI"])
fig, ax = plt.subplots(figsize=(7, 6))
if layer == "Класс RESTREND":
    cmap = ListedColormap(["#b42318", "#fdb022", "#e4e7ec", "#12b76a"])
    ax.imshow(z["map_class"], cmap=cmap, vmin=-2.5, vmax=1.5, extent=extent, interpolation="nearest")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, CLASS_LABELS.values())], fontsize=8, loc="lower left")
elif layer == "Тренд остатков":
    im = ax.imshow(z["map_resid_slope"] * 10, cmap="RdYlGn", vmin=-0.05, vmax=0.05, extent=extent)
    fig.colorbar(im, label="NDVI за 10 лет")
elif layer == "Тренд NDVI (без поправки)":
    im = ax.imshow(z["map_ndvi_slope"] * 10, cmap="RdYlGn", vmin=-0.05, vmax=0.05, extent=extent)
    fig.colorbar(im, label="NDVI за 10 лет")
elif layer == "Связь NDVI с осадками (R²)":
    im = ax.imshow(z["map_r2_precip"], cmap="viridis", vmin=0, vmax=1, extent=extent)
    fig.colorbar(im)
elif layer == "Год перелома":
    bi = z["map_break_index"]
    im = ax.imshow(np.where(np.isfinite(bi), years[0] + bi, np.nan), cmap="plasma", extent=extent)
    fig.colorbar(im, label="год")
elif layer == "ЦУР 15.3.1":
    ax.imshow(z["map_sdg_degraded"], cmap=ListedColormap(["#e4e7ec", "#b42318"]), extent=extent, interpolation="nearest")
else:
    im = ax.imshow(z["map_ndvi_mean"], cmap="YlGn", extent=extent)
    fig.colorbar(im)
ax.set_title(layer)

left, right = st.columns([1.2, 1])
left.pyplot(fig)
plt.close(fig)
ts = pd.DataFrame({"NDVI (среднее)": summary["mean_ndvi_by_year"].values(),
                   "Осадки, мм": summary["mean_precip_by_year"].values()}, index=[int(y) for y in summary["mean_ndvi_by_year"]])
right.subheader("Ряды по району")
right.line_chart(ts["NDVI (среднее)"])
right.bar_chart(ts["Осадки, мм"])
right.caption(f"Медианный R² связи NDVI с осадками: {summary['median_r2_ndvi_vs_precip']}. "
              "Чем он выше, тем важнее поправка на засуху.")

st.subheader("ЦУР 15.3.1 (упрощённый расчёт, «один против всех»)")
sdg = summary["sdg_15_3_1"]
st.table(pd.DataFrame({
    "Показатель": ["Деградировано, км²", "Улучшилось, км²", "Всего, км²", "Доля деградированных, %",
                   "— по продуктивности, %", "— по покрову, %", "— по почвенному углероду, %"],
    "Значение": [sdg["degraded_km2"], sdg["improved_km2"], sdg["total_km2"], sdg_value,
                 *sdg["by_subindicator_percent"].values()],
}))
st.subheader("Классы RESTREND, км²")
st.table(pd.DataFrame({"км²": summary["restrend_classes_km2"], "%": summary["restrend_classes_percent"]}))
if summary.get("break_years_top"):
    st.subheader("Годы переломов (число пикселей со значимым переломом)")
    st.bar_chart(pd.Series(summary["break_years_top"]).rename("пикселей"))

sb = RESULTS / f"{run}_salinity_burn.json"
if sb.exists():
    st.subheader("Засоление и гари")
    st.json(json.loads(sb.read_text(encoding="utf-8")))
fires = RESULTS / "fires_summary.csv"
if fires.exists():
    st.subheader("Пожары (NASA FIRMS)")
    f = pd.read_csv(fires)
    st.map(f.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
    st.dataframe(f.head(50))

st.download_button("Скачать сводку (JSON)", json.dumps(summary, ensure_ascii=False, indent=2), f"{run}_summary.json")
