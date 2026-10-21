"""Fixed Hermitian positive-definite approximations; original S stays unchanged."""
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import time
from ..errors import NumericalError


@dataclass
class PreconditionerData:
    kind: str
    blocks: list = field(default_factory=list)
    factor: object = None
    permutation: object = None
    scaling: object = None
    report: dict = field(default_factory=dict)
    _upper: object = field(default=None, repr=False)

    def apply(self,R):
        if self.kind=="ic0":
            p=self.permutation;z=self.scaling
            q=sp.linalg.spsolve_triangular(self.factor,(R*z[:,None])[p],lower=True)
            v=sp.linalg.spsolve_triangular(self._upper,q,lower=False)
            out=np.empty_like(R);out[p]=v
            return z[:,None]*out
        out=np.empty_like(R)
        for idx,L in self.blocks:
            out[idx]=la.cho_solve((L,True),R[idx],check_finite=False)
        return out


def prepare_cluster(S,partition):
    blocks=[]
    for block in partition["blocks"]:
        idx=np.asarray(block["orbitals"],dtype=np.int64)
        L=la.cholesky(S[idx][:,idx].toarray(),lower=True,check_finite=False)
        idx.setflags(write=False);L.setflags(write=False)
        blocks.append((idx,L))
    return PreconditionerData("cluster-jacobi",blocks=blocks,
        report=dict(block_count=len(blocks),block_sizes=[len(i) for i,L in blocks],
                    factor_bytes=sum(i.nbytes+L.nbytes for i,L in blocks)))


def ic0_reference(W,shift=0):
    """Small-matrix audit implementation, independent of the compiled factor."""
    lower=sp.tril(W,format="csr");lower.sort_indices()
    rows=[]
    for i in range(W.shape[0]):
        row={}
        for k in range(lower.indptr[i],lower.indptr[i+1]):
            j=int(lower.indices[k]);value=lower.data[k]
            if j<i:
                value-=sum(v*np.conj(rows[j].get(c,0)) for c,v in row.items() if c<j)
                row[j]=value/rows[j][j]
            else:
                pivot=(value+shift-sum(abs(v)**2 for v in row.values()))
                if not np.isfinite(pivot) or abs(pivot.imag)>1e-12 or pivot.real<=1e-14:
                    raise NumericalError(f"IC nonpositive pivot at row {i}")
                row[j]=np.sqrt(pivot.real)
        if i not in row:raise NumericalError("IC missing diagonal")
        rows.append(row)
    values=np.array([rows[i][j] for i in range(W.shape[0])
                     for j in lower.indices[lower.indptr[i]:lower.indptr[i+1]]],complex)
    return sp.csr_matrix((values,lower.indices.copy(),lower.indptr.copy()),shape=W.shape)


def prepare_ic0(S,*,ordering,shift_max):
    from .bounds import spd_bounds
    from ..validation import hermitian
    S=hermitian(S,"overlap")
    spd_bounds(S)  # A shift in M must never make an indefinite physical S admissible.
    return _prepare_ic0_validated(S,ordering=ordering,shift_max=shift_max)


def _prepare_ic0_validated(S,*,ordering,shift_max):
    from ..backends.native import require_extension,csr
    if ordering not in ("rcm","natural") or not np.isfinite(shift_max) or shift_max<0:
        raise NumericalError("invalid IC ordering/shift limit")
    start=time.perf_counter();z=1/np.sqrt(S.diagonal().real)
    W=(sp.diags(z)@S@sp.diags(z)).tocsr();W.eliminate_zeros();W.sort_indices()
    p=(sp.csgraph.reverse_cuthill_mckee(W,symmetric_mode=True) if ordering=="rcm"
       else np.arange(S.shape[0],dtype=np.int64))
    lower=sp.tril(W[p][:,p],format="csr");lower.sort_indices()
    extension=require_extension();native=csr(lower,1,extension);attempts=[]
    for shift in (0,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2):
        if shift>shift_max:continue
        try:
            values=extension.ic0_factor(native,shift)
        except (ValueError,RuntimeError) as exc:
            attempts.append(dict(shift=shift,status="FAILED",reason=str(exc)))
            continue
        attempts.append(dict(shift=shift,status="PASS"))
        L=sp.csr_matrix((values,lower.indices.copy(),lower.indptr.copy()),shape=S.shape)
        report=dict(ordering=ordering,shift=shift,shift_applies_to="M only, normalized coordinates",
                    attempts=attempts,nnz=L.nnz,factor_bytes=L.data.nbytes+L.indices.nbytes+L.indptr.nbytes,
                    minimum_pivot=float(np.min(L.diagonal().real**2)),seconds=time.perf_counter()-start)
        for a in (L.data,L.indices,L.indptr,z,p):a.setflags(write=False)
        import scipy
        report.update(permutation=p.tolist(),scipy_version=scipy.__version__)
        result=PreconditionerData("ic0",factor=L,permutation=p,scaling=z,report=report)
        result._upper=L.conj().T.tocsr()
        return result
    failure=NumericalError("IC(0) failed all allowed shifts; original S was not modified")
    failure.diagnostics=dict(attempts=attempts,ordering=ordering,shift_max=shift_max)
    raise failure
