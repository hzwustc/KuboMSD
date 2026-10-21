"""Serial scheduling and resource checks for independent complete LCAO operators."""
from dataclasses import dataclass, field
from contextlib import nullcontext
import os
import numpy as np
import scipy.sparse as sp
from ..contracts import ModelBundle
from ..errors import KuboMSDError, NumericalError
from ..validation import hermitian
from .assembly import complete_images
from .progress import BuildReport,environment_info
from .tables import prepare_tables, native_module, ALGORITHM


@dataclass
class BuilderOptions:
    operator_builder: str = 'cpp'
    tolerance: float = 1e-5
    table_settings: dict = field(default_factory=dict)
    threads: int = 1
    memory_limit_bytes: int = 2**31
    table_cache_dir: object = None
    progress_path: object = None
    build_distribution: str = 'serial'
    pair_chunk_size: int = 128
    mpi_io_batch_bytes: int = 64*2**20
    reference_settings: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.operator_builder not in ('reference','cpp','cpp-omp'): raise KuboMSDError('unknown operator_builder')
        if self.build_distribution not in ('serial','mpi'): raise KuboMSDError('unknown build_distribution')
        for k in ('threads','memory_limit_bytes','pair_chunk_size','mpi_io_batch_bytes'):
            v=getattr(self,k)
            if not isinstance(v,int) or isinstance(v,bool) or v<1: raise KuboMSDError(k+' must be a positive integer')
        if not np.isfinite(self.tolerance) or self.tolerance<=0: raise KuboMSDError('tolerance must be positive')
        if self.operator_builder=='cpp' and self.threads!=1: raise KuboMSDError('cpp is single-thread; use cpp-omp')
        if self.threads>len(os.sched_getaffinity(0)): raise KuboMSDError('threads exceed CPU affinity allocation')
        allocated=os.environ.get('SLURM_CPUS_PER_TASK')
        if allocated and self.threads>int(allocated): raise KuboMSDError('threads exceed SLURM_CPUS_PER_TASK')


@dataclass
class OperatorBuildResult:
    H_cov: object
    S: object
    composite: dict
    orbital_order: list
    atom_slices: list
    diagnostics: dict

    def as_model(self,provenance=None):
        return ModelBundle(self.H_cov,self.S,composite=self.composite,orbital_order=self.orbital_order,
                           atom_slices=self.atom_slices,provenance={**(provenance or {}),'operator_build':self.diagnostics})


def pair_tasks(structure,orbitals,memory_limit_bytes=2**31,*,reference=False):
    symbols=structure['symbols'];pos=np.asarray(structure['positions_A'],float);cell=np.asarray(structure['cell_A'],float)
    if pos.shape!=(len(symbols),3) or cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1e-12:
        raise KuboMSDError('invalid structure')
    species={s:i for i,s in enumerate(orbitals)}
    cuts=np.array([orbitals[s].cutoff_A for s in symbols])
    if reference:
        from itertools import product
        inv=np.linalg.inv(cell);keys=[];d=[]
        for i in range(len(symbols)):
            for j in range(i,len(symbols)):
                cutoff=cuts[i]+cuts[j];df=(pos[j]-pos[i])@inv;limits=cutoff*np.linalg.norm(inv,axis=0)
                ranges=[range(int(np.ceil(-df[q]-limits[q])),int(np.floor(-df[q]+limits[q]))+1) for q in range(3)]
                for R in product(*ranges):
                    if i==j and R<(0,0,0):continue
                    displacement=pos[j]+np.array(R)@cell-pos[i]
                    if np.linalg.norm(displacement)>=cutoff:continue
                    if (len(keys)+1)*64>memory_limit_bytes:raise KuboMSDError('reference neighbors exceed budget')
                    keys.append((i,j,*R));d.append(displacement)
        keys=np.asarray(keys,np.int64).reshape(-1,5);d=np.asarray(d,float).reshape(-1,3)
    else:keys,d=native_module().neighbors(pos,cell,np.linalg.inv(cell),cuts,memory_limit_bytes)
    atom_species=np.array([species[s] for s in symbols],np.int64)
    return keys,d,atom_species[keys[:,:2]]


def resource_estimate(keys,counts,options):
    # Full final CSR, canonical raw blocks, mirror COO and one stable-sort workspace.
    entries=sum(int(counts[i])*int(counts[j])*(1 if i==j and not np.any(k[2:]) else 2) for k in keys for i,j in [k[:2]])
    estimated=entries*400+(sum(counts)+1)*8*11
    if estimated>options.memory_limit_bytes: raise KuboMSDError(f'operator assembly estimated {estimated} bytes exceeds memory budget {options.memory_limit_bytes}')
    return int(estimated)


def thread_context():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1,user_api='blas')
    except ImportError:
        raise KuboMSDError('operator build requires threadpoolctl for bounded nested BLAS threads')


def get_tables(orbitals,options,report):
    with report.stage('table_build_or_load'):
        if options.table_cache_dir:
            from .cache import cached_tables
            table,meta=cached_tables(orbitals,options)
        else:
            from .tables import validated_tables
            table,meta=validated_tables(orbitals,options.tolerance,options.table_settings,options.memory_limit_bytes)
        report.details['table']=meta
    return table


def raw_blocks(keys,displacements,species,counts,orbitals,options,report,table=None):
    sizes=np.array([int(counts[i])*int(counts[j]) for i,j in keys[:,:2]],np.int64)
    offsets=np.r_[0,np.cumsum(sizes)];s=np.zeros(offsets[-1]);d=np.zeros((3,offsets[-1]))
    with report.stage('pair_evaluation'):
        for start in range(0,len(keys),options.pair_chunk_size):
            end=min(len(keys),start+options.pair_chunk_size)
            if options.operator_builder=='reference':
                from .reference import evaluate_reference_pair
                names=list(orbitals)
                for i in range(start,end):
                    a,b=[orbitals[names[v]] for v in species[i]]
                    ss,dd,diag=evaluate_reference_pair(a,b,displacements[i],{**options.reference_settings,'tolerance':options.tolerance})
                    if diag['status']!='PASS': raise NumericalError(f'REFERENCE_UNCONVERGED pair {i}: {diag}')
                    sl=slice(offsets[i],offsets[i+1]);s[sl]=ss.ravel();d[:,sl]=dd.reshape(3,-1)
            else:
                values=native_module().evaluate_pairs(table,dict(species=species[start:end],displacement_A=displacements[start:end],pair_ids=np.arange(start,end,dtype=np.int64)),options.threads)
                sl=slice(offsets[start],offsets[end]);s[sl]=values['S'];d[:,sl]=values['D']
            report.update(completed_pairs=end,total_pairs=len(keys))
    return dict(pair_ids=np.arange(len(keys),dtype=np.int64),offsets=offsets,S=s,D=d)


def assemble_result(structure,orbitals,h_images,keys,displacements,blocks,options,report):
    from .basis import convention_for,basis_map,transform_block,orbital_labels
    convention=convention_for(structure)
    maps={s:basis_map(o,convention) for s,o in orbitals.items()}
    report.details['orbital_convention']=convention
    counts=np.array([orbitals[s].n_orbitals for s in structure['symbols']],np.int64)
    atom_offsets=np.r_[0,np.cumsum(counts)];n=int(atom_offsets[-1]);native=None if options.operator_builder=='reference' else native_module()
    def csr(rows,cols,values):
        if native is None:return sp.coo_matrix((np.asarray(values,complex),(rows,cols)),shape=(n,n)).tocsr()
        data,indices,indptr=native.assemble_csr(np.asarray(rows,np.int64),np.asarray(cols,np.int64),np.asarray(values,complex),n,options.memory_limit_bytes)
        return sp.csr_matrix((data,indices,indptr),shape=(n,n))
    with report.stage('csr_assembly'):
        # Array-level block expansion, no Python scalar COO lists.
        reverse=np.array([i!=j or np.any(k[2:]) for k in keys for i,j in [k[:2]]])
        size=int(sum((2 if r else 1)*(blocks['offsets'][i+1]-blocks['offsets'][i]) for i,r in enumerate(reverse)))
        rows=np.empty(size,np.int64);cols=np.empty(size,np.int64);overlap=np.empty(size);xbar=np.empty((3,size));delta=np.empty((3,size))
        cursor=0
        for i,k in enumerate(keys):
            a,b=k[:2];shape=(counts[a],counts[b]);nr=int(np.prod(shape));rr,cc=np.indices(shape)
            s=blocks['S'][blocks['offsets'][i]:blocks['offsets'][i+1]].reshape(shape)
            d=blocks['D'][:,blocks['offsets'][i]:blocks['offsets'][i+1]].reshape((3,*shape))
            if convention!='abacus-real':
                ma,mb=maps[structure['symbols'][a]],maps[structure['symbols'][b]]
                s=transform_block(s,ma,mb);d=transform_block(d,ma,mb)
            v=displacements[i,:,None,None]
            sl=slice(cursor,cursor+nr);cursor+=nr
            rows[sl]=(rr+atom_offsets[a]).ravel();cols[sl]=(cc+atom_offsets[b]).ravel();overlap[sl]=s.ravel();xbar[:,sl]=d.reshape(3,-1);delta[:,sl]=(d-v*s).reshape(3,-1)
            if reverse[i]:
                sl=slice(cursor,cursor+nr);cursor+=nr
                rows[sl]=(cc+atom_offsets[b]).ravel();cols[sl]=(rr+atom_offsets[a]).ravel();overlap[sl]=s.ravel();xbar[:,sl]=(d-v*s).reshape(3,-1);delta[:,sl]=d.reshape(3,-1)
        S=csr(rows,cols,overlap)
        xbar_c=[csr(rows,cols,v) for v in xbar];delta_c=[csr(rows,cols,v) for v in delta]
        del rows,cols,overlap,xbar,delta
        images=complete_images(h_images,counts);hsize=sum(v.size for v in images.values())
        if hsize*160>options.memory_limit_bytes: raise KuboMSDError('Hamiltonian assembly exceeds memory budget')
        rows=np.empty(hsize,np.int64);cols=np.empty(hsize,np.int64);values=np.empty(hsize,complex);bond=np.empty((3,hsize),complex);cursor=0
        pos=np.asarray(structure['positions_A']);cell=np.asarray(structure['cell_A'])
        for (a,b,*R),v in sorted(images.items()):
            rr,cc=np.indices(v.shape);sl=slice(cursor,cursor+v.size);cursor+=v.size
            rows[sl]=(rr+atom_offsets[a]).ravel();cols[sl]=(cc+atom_offsets[b]).ravel();values[sl]=v.ravel()
            bond[:,sl]=(pos[b]+np.asarray(R)@cell-pos[a])[:,None]*v.ravel()
        H=csr(rows,cols,values);bc=[csr(rows,cols,v) for v in bond]
    with report.stage('matrix_validation'):
        H=hermitian(H,'H');S=hermitian(S,'S')
        from ..metric.bounds import spd_bounds
        bounds=spd_bounds(S)
        report.details['S_bounds']=bounds
    order=[f'atom{i}:{label}' for i,s in enumerate(structure['symbols']) for label in orbital_labels(orbitals[s],convention)]
    report.counts.update(n_atoms=len(counts),n_orbitals=n,unique_pairs=len(keys),S_nnz=S.nnz,H_nnz=H.nnz)
    return OperatorBuildResult(H,S,{a:(bc[q],xbar_c[q],delta_c[q]) for q,a in enumerate('xyz')},order,list(zip(atom_offsets[:-1].tolist(),atom_offsets[1:].tolist())),report.snapshot())


def build_operators(structure,orbitals,h_images,options=None):
    options=options or BuilderOptions()
    if options.build_distribution!='serial': raise KuboMSDError('use build_operators_mpi for MPI distribution')
    from ..parallel import check_serial_launcher
    # Imported parallel.py itself does not import mpi4py.
    check_serial_launcher()
    report=BuildReport(options.operator_builder,options.progress_path)
    report.details.update(algorithm=ALGORITHM,tolerance=options.tolerance,threads=options.threads,cpu_affinity=sorted(os.sched_getaffinity(0)),physical_operator_validation='UNVERIFIED')
    with thread_context():
        report.details['environment']=environment_info()
        with report.stage('neighbor_list'):
            keys,d,species=pair_tasks(structure,orbitals,options.memory_limit_bytes,reference=options.operator_builder=='reference')
            counts=[orbitals[s].n_orbitals for s in structure['symbols']]
            report.details['estimated_assembly_bytes']=resource_estimate(keys,counts,options)
        table=None if options.operator_builder=='reference' else get_tables(orbitals,options,report)
        blocks=raw_blocks(keys,d,species,counts,orbitals,options,report,table)
        result=assemble_result(structure,orbitals,h_images,keys,d,blocks,options,report)
    report.finish();result.diagnostics=report.snapshot();return result
