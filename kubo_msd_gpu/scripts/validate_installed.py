"""Run with the clean wheel interpreter, from outside the source tree."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
root=args.project.resolve();out=args.out.resolve()
out.mkdir(parents=True,exist_ok=False)
import kubo_msd
module=Path(kubo_msd.__file__).resolve()
assert module.is_relative_to(Path(sys.prefix)),module
assert not module.is_relative_to(root/'src')
assert not any(Path(v or '.').resolve()==root/'src' for v in sys.path)
requires=importlib.metadata.requires('kubo-msd-gpu')
assert not any('mpi4py' in r.lower() or 'rsatb' in r.lower() for r in requires)
kernels=sorted(str(p.relative_to(module.parent)) for p in (module.parent/'operators/gpu/kernels').glob('*.cu'))
assert len(kernels)==4
assert not list(module.parent.rglob('*.so'))
wheel=root/f'dist/kubo_msd_gpu-{kubo_msd.__version__}-py3-none-any.whl'
with zipfile.ZipFile(wheel) as z:
    assert len([n for n in z.namelist() if n.endswith('.cu')])==4
env={**os.environ,'CUPY_CACHE_DIR':str(out/'first_jit'),'PYTHONNOUSERSITE':'1',
     'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
env.pop('PYTHONPATH',None)
entry=Path(sys.prefix)/'bin/kubomsd-gpu';runs=[]
def run(label,argv):
    start=time.perf_counter()
    with (out/f'{label}.log').open('w') as f:
        subprocess.run([str(entry),*argv],cwd='/tmp',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    runs.append({'label':label,'wall_seconds':time.perf_counter()-start,'command':[str(entry),*argv]})
run('capabilities',['capabilities'])
run('cold_build',['prepare','--input',str(root/'examples/raw_lcao/input.json'),
                  '--out',str(out/'cache'),'--mode','build','--device-memory-limit-gb','2'])
build=json.loads((out/'cache/prepare_report.json').read_text())
assert build['table']['table_source']=='gpu_cold'
assert any(k['kernel']=='radial_transform' for k in build['gpu_pipeline']['kernel_calls'])
for method in ('exact','stepwise','spectral'):
    argv=['run','--input',str(out/'cache/model.json'),'--out',str(out/method),
          '--emin','-1','--emax','1','--n-energy','5','--tmax-fs','.02','--dt-fs','.01',
          '--memory-limit-gb','2','--device-memory-limit-gb','2','--axes','x,y',
          '--kernel','jackson','--n-moments','12','--spectral-min','-10','--spectral-max','10']
    if method!='exact':
        argv+=['--solver','stochastic-kpm','--stochastic-method',method,'--nsamples','3',
               '--sample-batch','2','--statistics-block','2','--integration-dt-fs','.01']
    run(method,argv)
    meta=json.loads((out/method/'metadata.json').read_text())
    assert meta['run_status']=='complete' and meta['backend_used']=='gpu'
report={'status':'PASS','module':str(module),'prefix':sys.prefix,'cwd':str(Path.cwd()),
        'pythonpath':env.get('PYTHONPATH'),'version':kubo_msd.__version__,'requirements':requires,
        'cuda_sources':kernels,'runs':runs,
        'capabilities':json.loads((out/'capabilities.log').read_text())['capabilities'],
        'first_JIT':'empty dedicated directory; cold radial/assembly kernels actually executed',
        'cpu_native_extension':False,'system_site_packages':False}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
