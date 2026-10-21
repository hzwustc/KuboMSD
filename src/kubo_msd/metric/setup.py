"""Immutable host S and configuration-bound runtime initialization, never a global cache."""
from dataclasses import dataclass, field
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from ..validation import hermitian,matrix_fingerprint
from ..errors import NumericalError
from .bounds import spd_bounds


def setup_key(S,config,atom_slices):
    import hashlib
    from pathlib import Path
    partition=None if config.metric_partition_file is None else hashlib.sha256(Path(config.metric_partition_file).read_bytes()).hexdigest()
    return matrix_fingerprint(sp.csr_matrix(S,dtype=np.complex128))+repr((config.preconditioner,config.metric_solver,config.backend if config.metric_solver=='cholesky' else None,atom_slices,config.sampling_tol,config.sampling_max_order,config.memory_limit_gb if config.preconditioner=='full-cholesky' or config.metric_solver=='cholesky' else None,partition,config.metric_ic_ordering,config.metric_ic_shift_max,'spd-v2'))


@dataclass
class HostMetricSetup:
    S: object
    key: str
    diagonal: object
    bounds: tuple
    blocks: list
    seconds: float
    direct_factor: object = field(default=None,repr=False)
    preconditioner_data: object = field(default=None,repr=False)
    metric_identity: object = field(default=None,repr=False)
    factor_seconds: float = 0.
    _w_bounds: object = field(default=None,repr=False)

    def validate(self,S,config,atom_slices):
        if setup_key(S,config,atom_slices)!=self.key:
            raise NumericalError('host metric setup identity/configuration mismatch')
        if setup_key(self.S,config,atom_slices)!=self.key:
            raise NumericalError('host metric setup was mutated')

    def normalized_bounds(self):
        if self._w_bounds is None:
            z=1/np.sqrt(self.diagonal);W=sp.diags(z)@self.S@sp.diags(z)
            self._w_bounds=spd_bounds(W)
        return self._w_bounds


def prepare_host_metric(host_S,config,atom_slices=None,*,metric_identity=None):
    start=time.perf_counter();S=hermitian(host_S,'overlap');blocks=[];direct_factor=None;factor_seconds=0.
    if config.metric_solver=='cholesky':
        from .cholesky import check_budget
        check_budget(S,config)
    if config.preconditioner=='full-cholesky' or (config.metric_solver=='cholesky' and config.backend=='cpu'):
        n=S.shape[0]
        # Reserve dense input, factor and LAPACK/copy workspace before toarray.
        if 3*16*n*n>config.memory_limit_gb*1024**3:
            raise NumericalError('full-cholesky factor/workspace estimate exceeds memory budget')
        try:
            factor_start=time.perf_counter()
            L=la.cholesky(S.toarray(order='F'),lower=True,overwrite_a=True,check_finite=False)
            factor_seconds=time.perf_counter()-factor_start
        except la.LinAlgError as exc:
            raise NumericalError('overlap is not positive definite (Cholesky factorization)') from exc
        upper=float(np.asarray(abs(S).sum(axis=0)).max())
        rcond,info=la.get_lapack_funcs('pocon',(L,))(L,upper,uplo='L')
        if info!=0 or not np.isfinite(rcond) or rcond<=0:
            raise NumericalError('Cholesky condition estimation failed')
        bounds=(upper*rcond,upper,{'source':'LAPACK xPOCON inverse 1-norm estimate',
                                  'certified':False,'positive_definiteness':'Cholesky factorization'})
        if config.metric_solver=='cholesky': direct_factor=L
        else: blocks=[(0,n,L)]
    else:
        bounds=spd_bounds(S,allow_dense=False) if config.metric_solver=='cholesky' and config.backend=='gpu' else spd_bounds(S)
    key=setup_key(S,config,atom_slices);diagonal=S.diagonal().real.copy()
    if config.preconditioner=='block-jacobi':
        slices=atom_slices or [(j,j+1) for j in range(S.shape[0])]
        if [i for a,b in slices for i in range(a,b)]!=list(range(S.shape[0])):raise NumericalError('atom_slices must partition orbitals')
        for a,b in slices:blocks.append((a,b,la.cholesky(S[a:b,a:b].toarray(),lower=True)))
    data=None
    if config.preconditioner=='cluster-jacobi':
        from .partition import load_metric_partition
        from .preconditioners import prepare_cluster
        if not metric_identity or not all(metric_identity.values()):
            raise NumericalError('cluster-Jacobi requires explicit model identity')
        partition=load_metric_partition(config.metric_partition_file,s_fingerprint=matrix_fingerprint(S),
                                        atom_slices=atom_slices,n_orbitals=S.shape[0],**metric_identity)
        data=prepare_cluster(S,partition)
    if config.preconditioner=='ic0':
        from .preconditioners import _prepare_ic0_validated
        data=_prepare_ic0_validated(S,ordering=config.metric_ic_ordering,shift_max=config.metric_ic_shift_max)
    for array in (S.data,S.indices,S.indptr,diagonal):array.setflags(write=False)
    for a,b,L in blocks:L.setflags(write=False)
    if direct_factor is not None:direct_factor.setflags(write=False)
    return HostMetricSetup(S,key,diagonal,bounds,blocks,time.perf_counter()-start,direct_factor,data,metric_identity,factor_seconds)
