"""Synthetic 1e-5/1e-6/1e-7 operator-to-transport convergence, not material acceptance."""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.input import parse_build_input
from kubo_msd.operators.gpu.builder import build_gpu_operators
from kubo_msd.operators.gpu.contracts import GpuBuildOptions
from kubo_msd.operators.gpu.runtime import make_device_operators
from kubo_msd.config import RunConfig
from kubo_msd.solvers.exact import iter_exact_blocks
from kubo_msd.solvers.stochastic import iter_stochastic_blocks

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
parsed=parse_build_input(ROOT/'examples/raw_lcao/input.json')
ctx=create_gpu_context(0,2*1024**3);rows=[];previous={}
for tol in (1e-5,1e-6,1e-7):
    result=build_gpu_operators(parsed,GpuBuildOptions(integral_tolerance=tol),ctx)
    row={'tolerance':tol,'settings':result.report['table']['settings'],
         'condition_upper_bound':result.report['metric_bounds'][1]/result.report['metric_bounds'][0],
         'table_convergence':result.report['table']['convergence'],'transport':{}}
    for method in ('exact','stepwise','spectral'):
        cfg=RunConfig(np.linspace(-1,1,9),[0,.02,.04],backend='gpu',
            solver='exact' if method=='exact' else 'stochastic-kpm',kernel='jackson',
            spectral_bounds=(-10,10),n_moments=16,axes=('x','y','z'),seed=1729,
            nsamples=5,statistics_block=2,sample_batch=2,integration_dt_fs=.02,segment_fs=.02)
        ops=make_device_operators(result.model,cfg,ctx)
        blocks=list(iter_exact_blocks(result.model,ops,cfg) if method=='exact'
                    else iter_stochastic_blocks(ops,cfg,method))
        values={k:v for k,v in blocks[0].arrays.items() if k.startswith(
            ('msd_A2/','diffusion_A2_per_fs/','vac_A2_per_fs2/','numerator_msd',
             'numerator_diffusion','numerator_vac'))}
        comparisons=[]
        if method in previous:
            for k,v in values.items():
                old=previous[method][k]
                np.testing.assert_array_equal(np.isfinite(old),np.isfinite(v),err_msg=k)
                np.testing.assert_allclose(v,old,rtol=.01,atol=1e-8,equal_nan=True,err_msg=k)
                good=np.isfinite(v)
                comparisons.append({'quantity':k,'max_abs_change':float(np.max(abs(v[good]-old[good]),initial=0)),
                                    'valid_nodes':int(good.sum()),'status':'PASS'})
        previous[method]=values
        np.savez_compressed(a.out/f'{tol:g}_{method}.npz',**values)
        row['transport'][method]={'status':'PASS' if comparisons else 'reference level',
                                  'comparisons':comparisons}
    rows.append(row)
    (a.out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
assert len({json.dumps(r['settings'],sort_keys=True) for r in rows})==3
(a.out/'report.json').write_text(json.dumps({'status':'PASS','scope':'synthetic two-orbital complete construction and transport',
    'rtol':.01,'atol_original_units':1e-8,'rows':rows,'gpu_pipeline':ctx.report(),
    'material_tolerance_convergence':'NOT RUN'},indent=2)+'\n')
print('PASS: three different table settings; EXACT, stepwise and spectral adjacent-level M/D/VAC')
