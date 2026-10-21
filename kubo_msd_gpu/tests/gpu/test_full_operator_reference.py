from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.input import parse_build_input
from kubo_msd.operators.gpu.builder import build_gpu_operators
from kubo_msd.operators.gpu.contracts import GpuBuildOptions

@pytest.mark.parametrize('label',['raw','tilted'])
def test_full_matrices_match_frozen_cpu_and_translation(label,tmp_path):
    root=Path(__file__).parents[2]
    ref=np.load(root/f'tests/data/cpu_{label}_operators.npz')
    p=parse_build_input(root/'examples/raw_lcao/input.json')
    p=replace(p,structure={**p.structure,'cell_A':ref['cell_A'],'positions_A':ref['positions_A']})
    ctx=create_gpu_context(0,1024**3)
    options=GpuBuildOptions(table_cache_dir=tmp_path/'tables')
    for shift in (0.,np.array([1.3,-.7,.81])):
        moved=replace(p,structure={**p.structure,'positions_A':p.structure['positions_A']+shift})
        result=build_gpu_operators(moved,options,ctx)
        got={'H':result.model.H_cov,'S':result.model.S}
        for a,parts in result.model.composite.items():
            for name,m in zip(('bond','xbar','delta'),parts):got[a+'_'+name]=m
        for name,mat in got.items():
            np.testing.assert_allclose(ctx.to_host(mat.toarray()),ref[name],rtol=1e-10,atol=1e-12)
