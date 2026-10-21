"""One shared GPU constructor. No solver, seed, energy or time input."""
import hashlib
import json
import numpy as np
from .contracts import DeviceModelBundle,GpuBuildResult
from .plan import make_pair_plan
from .tables import build_tables_gpu
from .pairs import evaluate_pairs_gpu
from .assembly import pair_coo,h_coo,stable_csr
from .basis import align_integral_coo
from ...metric.gpu_setup import hermitian_gpu,spd_bounds_gpu


def build_gpu_operators(parsed,options,context):
    if options.device_memory_limit_bytes is not None:
        context.memory_limit_bytes=min(context.memory_limit_bytes,options.device_memory_limit_bytes)
    with context.phase('cpu_plan'):plan=make_pair_plan(parsed)
    with context.activate():
        tables=build_tables_gpu(parsed,options,context)
        identity=hashlib.sha256(tables.identity.encode())
        identity.update(('operator-basis-v2:'+parsed.harmonic_convention).encode())
        for name in ('positions_A','cell_A'):identity.update(np.asarray(parsed.structure[name]).tobytes())
        identity.update(json.dumps(parsed.structure['symbols']).encode())
        for key,v in sorted(parsed.h_images.items()):identity.update(np.asarray(key,np.int64).tobytes());identity.update(v.tobytes())
        identity=identity.hexdigest()
        count=int(plan.offsets[-1])*2
        # Full compact COO, CSR, stable sort, concatenation and all 9 composites.
        context.require_bytes(count*(16*7*3+8*12),'full compact assembly working set')
        chunks=[];batch=options.pair_batch_size
        for start in range(0,len(plan.pair_ids),batch):
            b=plan.batch(start,min(start+batch,len(plan.pair_ids)))
            blocks=evaluate_pairs_gpu(tables,b,context)
            chunks.append(pair_coo(blocks,parsed,context))
            del blocks
            if options.progress_path:
                from pathlib import Path
                Path(options.progress_path).write_text(json.dumps({'phase':'pairs','completed':min(start+batch,len(plan.pair_ids)),'total':len(plan.pair_ids)})+'\n')
        n=len(parsed.orbital_order);xp=context.xp
        if chunks:
            rows=xp.concatenate([c[0] for c in chunks]);cols=xp.concatenate([c[1] for c in chunks]);seq=xp.concatenate([c[2] for c in chunks]);values=xp.concatenate([c[3] for c in chunks],axis=1)
            del chunks
            rows,cols,values=align_integral_coo(rows,cols,values,parsed,context)
            mats=stable_csr(rows,cols,seq,values,n,context)
            del rows,cols,seq,values
        else:raise ValueError('no orbital support pairs')
        h=h_coo(parsed,context);hmats=stable_csr(*h,n,context);del h
        composite={axis:(hmats[1+a],mats[1+a],mats[4+a]) for a,axis in enumerate('xyz')}
        with context.phase('device_validation'):
            hermitian_gpu(hmats[0],'H',context);hermitian_gpu(mats[0],'S',context)
            lo,hi,spd=spd_bounds_gpu(mats[0],context)
        provenance={'files':list(parsed.input_hashes),'input_format':'legacy-lcao-deeptb','builder_used':'gpu',
                    'algorithm':tables.metadata['algorithm'],'table_identity':tables.identity,
                    'table_source':tables.metadata['table_source'],'integrals':tables.metadata['convergence'],
                    'internal_units':['eV','fs','angstrom'],'dipole_origin':'bra center before composite assembly',
                    'hamiltonian_orbital_convention':parsed.harmonic_convention,
                    'integral_internal_convention':'abacus-real',
                    'basis_alignment':'operator-basis-v2; S/dipole aligned on device; H unchanged'}
        model=DeviceModelBundle(hmats[0],mats[0],context.device_id,composite=composite,
            orbital_order=parsed.orbital_order,atom_slices=parsed.atom_slices,provenance=provenance,cache_identity=identity)
        report={'builder_used':'gpu','cache_identity':identity,'table':tables.metadata,'pairs':plan.report,
                'n_orbitals':n,'nnz':{'H':model.H_cov.nnz,'S':model.S.nnz},'metric_bounds':[lo,hi],'spd':spd}
        return GpuBuildResult(model,report)
