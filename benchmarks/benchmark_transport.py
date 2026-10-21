#!/usr/bin/env python3
"""Reproducible end-to-end small-model timings; never infer a GPU speedup."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from kubo_msd.config import RunConfig
from kubo_msd.dispatch import run_input
from kubo_msd.io.results import read_results

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True,type=Path); p.add_argument('--quick',action='store_true'); p.add_argument('--input',type=Path); p.add_argument('--backend',choices=['cpu','gpu'],default='cpu'); a=p.parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=False)
    model=a.input or Path(__file__).resolve().parents[1]/'examples/nonorthogonal/model.json'
    cfg=RunConfig(energy_eV=np.linspace(-.2,.3,3 if a.quick else 31),time_fs=np.linspace(0,.2 if a.quick else 1,3 if a.quick else 21),axes=('x','y'),backend=a.backend,kernel='jackson',n_moments=12 if a.quick else 64,nsamples=8 if a.quick else 64,sample_batch=4,statistics_block=1,integration_dt_fs=.01)
    runs=[]; reference=None
    for method in ('exact','stepwise','spectral'):
        c=cfg if method=='exact' else replace(cfg,solver='stochastic-kpm',stochastic_method=method,include_vac=method=='spectral')
        t=time.perf_counter(); summary=run_input(model,c,a.out/method); elapsed=time.perf_counter()-t; data=read_results(a.out/method)
        estimator=data['metadata']['primary_estimator']; values=data['arrays']['msd_A2/'+estimator]
        if reference is None: reference=values
        finite=np.isfinite(values)&np.isfinite(reference)
        error=float(np.max(abs(values[finite]-reference[finite]))) if finite.any() else None
        runs.append({'method':method,'backend':a.backend,'wall_seconds':elapsed,'max_absolute_msd_error_A2_against_exact':error,'finite_compared_points':int(finite.sum()),'phases':summary.diagnostics,'initialization':data['metadata']['initialization']})
    report={'claim':'small-model measurements only; no material/GPU speedup claim','accuracy_matched':False,'reason':'finite-sample and integration errors reported; no matched-error calibration inferred','input':str(model.resolve()),'resolved_base_config':cfg.as_dict(),'runs':runs}
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(a.out/'report.json')

if __name__=='__main__': main()
