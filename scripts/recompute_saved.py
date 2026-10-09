"""Recompute maps from saved inputs without fetching or inventing satellite data."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from steppewatch.pipeline import analyze,save,plot

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path);p.add_argument('--pixel-km2',type=float,required=True)
    p.add_argument('--baseline',type=int,default=15);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--restore-synthetic-labels',action='store_true')
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as z:
        years,ndvi,precip=(z[k].copy() for k in ('years','ndvi','precip'))
        extent=z['extent'].tolist() if 'extent' in z else None
    extra={}
    if a.restore_synthetic_labels:
        from steppewatch.synthetic import make_cube
        c=make_cube()
        if ndvi.shape!=c['ndvi'].shape or not np.array_equal(years,c['years']) or not np.allclose(ndvi,c['ndvi'],atol=1e-7) or not np.allclose(precip,c['precip'],atol=1e-4):
            p.error('Saved inputs do not match the deterministic synthetic fixture')
        extra={k:c[k] for k in ('lc_start','lc_end','soc_start','soc_end')}
    result=analyze(years,ndvi,precip,a.pixel_km2,a.baseline,**extra)
    result['summary']['recomputed_from']=a.input.name
    result['summary']['data_kind']='synthetic' if extra else 'saved_inputs_unverified_ground_truth'
    # Previous products are available in Git history; preserve only original input arrays.
    save(result,a.out,a.name,years,ndvi,precip,extent)
    plot(result,a.out/f'{a.name}.png',f'{a.name}: revision 2 (FDR; ground verification required)',extent)
    print(json.dumps(result['summary']['sdg_15_3_1'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
