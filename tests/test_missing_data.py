import numpy as np
import pytest
from steppewatch.sdg import land_cover_change, soc_change, sdg_15_3_1, productivity
from steppewatch.trends import fdr_bh

def test_unknown_cover_is_not_forest():
    assert land_cover_change(np.array([0,np.nan,2]),np.array([6,6,6])).tolist()==[-9,-9,-1]

def test_missing_carbon_is_not_stable():
    assert soc_change(np.array([0,np.nan,30]),np.array([1,2,26])).tolist()==[-9,-9,-1]

def test_incomplete_sdg_is_only_screening():
    r=sdg_15_3_1(np.array([-1,0]),None,None)
    assert r['sdg_15_3_1_percent'] is None and r['coverage_percent']==0
    assert r['screening_degraded_percent']==50

def test_partial_coverage_is_reported():
    r=sdg_15_3_1(np.array([-1,0]),np.array([0,-9]),np.array([0,0]))
    assert r['sdg_15_3_1_percent'] is None and r['coverage_percent']==50

def test_baseline_and_areas():
    with pytest.raises(ValueError):productivity(np.ones((10,2)),9)
    with pytest.raises(ValueError):sdg_15_3_1(np.array([0]),None,None,pixel_area_km2=-1)

def test_incomplete_productivity_is_unknown():
    ndvi=np.ones((10,2));ndvi[2,0]=np.nan
    assert productivity(ndvi,5)['lpd'][0]==-9

def test_fdr_adjustment():
    q=fdr_bh(np.array([.01,.04,.03,np.nan]))
    np.testing.assert_allclose(q[:3],[.03,.04,.04])
    assert np.isnan(q[3])
