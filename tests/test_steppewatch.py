import numpy as np
import pytest

from steppewatch.breaks import single_break
from steppewatch.indices import dnbr_severity, salinity_class, salinity_indices
from steppewatch.piosphere import distance_to_points, fit_piosphere
from steppewatch.sdg import land_cover_change, productivity, sdg_15_3_1, soc_change
from steppewatch.synthetic import make_cube
from steppewatch.trends import mann_kendall, restrend


def test_mann_kendall_detects_trend_and_noise():
    rng = np.random.default_rng(0)
    t = np.arange(25)
    y = np.column_stack([0.01 * t + rng.normal(0, 0.02, 25), rng.normal(0, 0.02, (25, 400))])
    r = mann_kendall(y)
    assert r["p"][0] < 0.01 and r["slope"][0] == pytest.approx(0.01, abs=0.003)
    false_alarms = np.mean(r["p"][1:] < 0.05)  # для чистого шума ≈ α
    assert 0.01 < false_alarms < 0.10


@pytest.fixture(scope="module")
def cube():
    return make_cube(n=120)


def test_restrend_separates_drought_from_degradation(cube):
    c = cube
    r = restrend(c["ndvi"], c["precip"])
    t, cls = c["truth"], r["class"]
    assert np.mean(cls[t == 6] == -1) > 0.8  # аридизация → «засуха», не деградация
    assert np.mean(cls[t == 1] == -2) > 0.8  # засоление → деградация
    assert np.mean(cls[t == 2] == -2) > 0.7  # перевыпас → деградация
    assert np.mean(cls[t == 0] == -2) < 0.1  # ложные тревоги на уровне α


def test_single_break_finds_year():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(12), -0.1 * np.ones(13)] + rng.normal(0, 0.01, 25)
    r = single_break(y)
    assert abs(r["break"] - 12) <= 1 and r["p"] < 0.01


def test_sdg_one_out_all_out():
    lpd = np.array([[-1, 0], [0, 1]])
    lc = np.array([[0, -1], [0, 0]])
    soc = np.array([[0, 0], [0, 0]])
    r = sdg_15_3_1(lpd, lc, soc, 1.0)
    assert r["sdg_15_3_1_percent"] == 50.0
    assert land_cover_change(np.array([2]), np.array([6]))[0] == -1  # степь → голые земли
    assert soc_change(np.array([30.0]), np.array([26.0]))[0] == -1


def test_productivity_flags_declining_pixels(cube):
    p = productivity(cube["ndvi"], 15)
    assert (p["lpd"][cube["truth"] == 1] == -1).mean() > 0.5
    assert (p["lpd"][cube["truth"] == 0] == -1).mean() < 0.15


def test_salinity_and_dnbr(cube):
    b = cube["bands"]
    idx = salinity_indices(b["blue"], b["green"], b["red"], b["nir"], b["swir1"], b["swir2"])
    sal = salinity_class(idx, cube["ndvi"][-1])
    assert sal[cube["truth"] == 1].mean() > sal[cube["truth"] == 0].mean()
    d, sev = dnbr_severity(cube["nbr_pre"], cube["nbr_post"])
    assert (sev[cube["truth"] == 4] >= 2).mean() > 0.9


def test_piosphere_radius(cube):
    n = cube["ndvi"].shape[1]
    dist = distance_to_points((n, n), cube["pixel_km"], [(35, 40)])
    fit = fit_piosphere(cube["ndvi"][-1], dist, max_km=8)
    assert fit["ok"] and 0.5 < fit["overgrazing_radius_km"] < 8
