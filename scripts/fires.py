"""Активные очаги степных пожаров (NASA FIRMS) по району: таблица очагов и сводка по пожарам.

    export FIRMS_MAP_KEY=...            # бесплатный ключ: firms.modaps.eosdis.nasa.gov/api/map_key
    python scripts/fires.py --aoi data/aoi/betpakdala.geojson --days 7
    python scripts/fires.py --bbox 46 40 88 56 --days 1          # весь Казахстан, последние сутки
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from steppewatch.fires import SOURCES, fire_summary, hotspots  # noqa: E402
from steppewatch.pipeline import load_aoi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi")
    ap.add_argument("--bbox", type=float, nargs=4)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--date", help="начальная дата YYYY-MM-DD (архив)")
    ap.add_argument("--source", default="VIIRS_SNPP_NRT", choices=SOURCES)
    a = ap.parse_args()
    bbox = tuple(a.bbox) if a.bbox else load_aoi(a.aoi)[1]
    frames = []
    # Длинные периоды запрашиваются кусками по 10 суток (ограничение API).
    start = pd.Timestamp(a.date) if a.date else None
    left = a.days
    while left > 0:
        n = min(left, 10)
        frames.append(hotspots(bbox, n, a.source, date=start.strftime("%Y-%m-%d") if start is not None else None))
        if start is None:
            break
        start += pd.Timedelta(days=n)
        left -= n
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if df.empty:
        print("Очагов не обнаружено")
        return
    df.to_csv(out / "fires_hotspots.csv", index=False)
    s = fire_summary(df)
    s.to_csv(out / "fires_summary.csv", index=False)
    print(f"Очагов: {len(df)}, пожаров (кластеров): {len(s)}")
    print(s.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
