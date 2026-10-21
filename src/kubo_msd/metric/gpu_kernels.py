"""Lazy CUDA primitives and bounded host polling of per-column device state."""
import numpy as np
from .residual import assess_residual


class DeviceBlocks:
    """One cooperative double-triangular launch per distinct block size.

    Each CUDA block owns one factor/RHS column. The disjoint orbital partition
    makes its gather, row updates and scatter independent of all other blocks.
    Only the current pivot value uses shared memory, so large valid clusters
    are handled by strided rows without a block-size/shared-memory cutoff.
    """
    def __init__(self,blocks,backend):
        self.xp=backend.xp;self.groups=[];self.applications=0;self.kernel_launches=0
        self.n=sum(len(indices) for indices,factor in blocks)
        grouped={}
        for indices,factor in blocks:
            grouped.setdefault(len(indices),[]).append((indices,factor))
        for size,entries in sorted(grouped.items()):
            indices=backend.asarray(np.stack([i for i,L in entries]).astype(np.int64),
                                    name="grouped block indices",phase="host_metric_setup")
            factors=backend.asarray(np.ascontiguousarray(np.stack([L for i,L in entries]),
                                                        dtype=np.complex128),
                                    name="grouped block factors",phase="host_metric_setup")
            self.groups.append((size,len(entries),indices,factors))
        self.kernel=self.xp.RawKernel(r'''
        #include <cupy/complex.cuh>
        extern "C" __global__ void block_triangular(
            const complex<double>* factors,const long long* indices,
            const complex<double>* rhs,complex<double>* out,int m,int b) {
            long long group=blockIdx.x/b;
            int column=blockIdx.x%b;
            const complex<double>* L=factors+group*(long long)m*m;
            const long long* ids=indices+group*m;
            __shared__ double pivot_real,pivot_imag;
            for(int row=threadIdx.x;row<m;row+=blockDim.x)
                out[ids[row]*b+column]=rhs[ids[row]*b+column];
            __syncthreads();
            for(int i=0;i<m;++i) {
                if(threadIdx.x==0) {
                    complex<double> value=out[ids[i]*b+column]/L[(long long)i*m+i];
                    out[ids[i]*b+column]=value;
                    pivot_real=value.real();pivot_imag=value.imag();
                }
                __syncthreads();
                complex<double> value(pivot_real,pivot_imag);
                for(int row=threadIdx.x;row<m;row+=blockDim.x)
                    if(row>i)out[ids[row]*b+column]-=L[(long long)row*m+i]*value;
                __syncthreads();
            }
            for(int i=m-1;i>=0;--i) {
                if(threadIdx.x==0) {
                    complex<double> value=out[ids[i]*b+column]/conj(L[(long long)i*m+i]);
                    out[ids[i]*b+column]=value;
                    pivot_real=value.real();pivot_imag=value.imag();
                }
                __syncthreads();
                complex<double> value(pivot_real,pivot_imag);
                for(int row=threadIdx.x;row<m;row+=blockDim.x)
                    if(row<i)out[ids[row]*b+column]-=conj(L[(long long)i*m+row])*value;
                __syncthreads();
            }
        }''',"block_triangular")

    def apply(self,R,out=None):
        R=self.xp.ascontiguousarray(R,dtype=self.xp.complex128)
        if R.ndim!=2 or R.shape[0]!=self.n or R.shape[1]<1:
            raise ValueError("grouped triangular RHS shape must be (N, positive batch)")
        if out is None:out=self.xp.empty_like(R)
        elif (not isinstance(out,self.xp.ndarray) or out.shape!=R.shape or
              out.dtype!=R.dtype or not out.flags.c_contiguous):
            raise ValueError("grouped triangular output workspace shape/layout mismatch")
        if R.data.ptr<out.data.ptr+out.nbytes and out.data.ptr<R.data.ptr+R.nbytes:
            raise ValueError("grouped triangular RHS and output workspace must not overlap")
        for size,count,indices,factors in self.groups:
            threads=min(256,max(32,1<<(size-1).bit_length()))
            self.kernel((count*R.shape[1],),(threads,),
                        (factors,indices,R,out,np.int32(size),np.int32(R.shape[1])))
            self.kernel_launches+=1
        self.applications+=1
        return out

    def diagnostics(self):
        return dict(groups={str(size):count for size,count,indices,L in self.groups},
                    applications=self.applications,kernel_launches=self.kernel_launches,
                    execution="one fused double-triangular kernel per block-size group",
                    scope="explicit grouped preconditioner kernels; other/library kernels excluded")


class DeviceIC:
    def __init__(self,data,backend):
        self.backend=backend;self.xp=backend.xp
        self.L=backend.matrix(data.factor,name="IC factor")
        self.upper=self.L.conj().T.tocsr()
        backend._record("D2D",self.upper,"IC adjoint factor","metric_factor","sparse triangular application")
        self.p=backend.asarray(data.permutation,name="IC permutation")
        self.z=backend.asarray(data.scaling,name="IC scaling")

    def apply(self,R):
        from cupyx.scipy.sparse.linalg import spsolve_triangular
        first=spsolve_triangular(self.L,(R*self.z[:,None])[self.p],lower=True)
        second=spsolve_triangular(self.upper,first,lower=False)
        out=self.xp.empty_like(R);out[self.p]=second
        return self.z[:,None]*out


class DevicePCG:
    def __init__(self,metric):
        self.metric=metric;self.xp=metric.xp
        # One conditional sparse product checks candidate convergence without
        # reading a scalar on the host or doing a full SpMM on unchecked columns.
        index="int" if metric.S.indices.dtype==np.int32 else "long long"
        code=r'''
        #include <cupy/complex.cuh>
        extern "C" __global__ void residual(
            const INDEX* ptr,const INDEX* ids,const complex<double>* data,
            const complex<double>* q,const complex<double>* y,
            const bool* check,complex<double>* out,long long n,int b) {
            long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
            if(t>=n*b)return;
            int j=t%b;long long i=t/b;
            if(!check[j]){out[t]=complex<double>(0,0);return;}
            complex<double> value=q[t];
            for(INDEX k=ptr[i];k<ptr[i+1];++k)value-=data[k]*y[(long long)ids[k]*b+j];
            out[t]=value;
        }'''.replace("INDEX",index)
        self.residual=self.xp.RawKernel(code,"residual")
        self.workspace=None

    def solve(self,Q,threshold,cfg):
        xp=self.xp;m=self.metric;n,b=Q.shape
        if self.workspace is None or self.workspace[0].shape!=Q.shape:
            self.workspace=[xp.empty_like(Q) for _ in range(5)]
        Y,R,P,truth,Zwork=self.workspace
        Y.fill(0);R[:]=Q
        norm=lambda a:xp.sqrt(xp.sum(xp.abs(a)**2,axis=0))
        active=norm(Q)>threshold
        Z=m.precondition(R,out=Zwork);P[:]=xp.where(active[None,:],Z,0)
        rz=xp.sum(R.conj()*Z,axis=0).real
        iters=xp.zeros(b,dtype=xp.int64);broken=xp.zeros(b,dtype=bool)
        checks=xp.zeros((),dtype=xp.int64);restarts=xp.zeros((),dtype=xp.int64)
        for k in range(cfg.metric_maxiter):
            SP=m.S@P;curvature=xp.sum(P.conj()*SP,axis=0)
            bad=active & (~xp.isfinite(curvature) | ~xp.isfinite(rz) | (rz<=0) |
                (curvature.real<=0) | (xp.abs(curvature.imag)>1e-10*xp.maximum(xp.abs(curvature.real),xp.finfo(float).tiny)))
            broken |= bad;active &= ~bad
            alpha=xp.where(active,rz/xp.where(active,curvature.real,1),0)
            # Inactive columns remain bitwise frozen even after a breakdown.
            Y[:]=xp.where(active[None,:],Y+P*alpha,Y)
            R[:]=xp.where(active[None,:],R-SP*alpha,R)
            iters+=active
            bad=active & (~xp.all(xp.isfinite(Y),axis=0) | ~xp.all(xp.isfinite(R),axis=0))
            broken |= bad;active &= ~bad
            rn=norm(R);periodic=(k+1)%cfg.metric_check_interval==0
            check=active & (True if periodic else rn<=threshold)
            # Fixed policy only refreshes on its historical interval.
            if cfg.metric_residual_policy=="fixed" and not periodic:check=xp.zeros_like(active)
            self.residual(((n*b+127)//128,),(128,),
                (m.S.indptr,m.S.indices,m.S.data,Q,Y,check,truth,np.int64(n),np.int32(b)))
            checks+=xp.any(check)
            bad=check & ~xp.all(xp.isfinite(truth),axis=0);broken|=bad;active&=~bad;check&=~bad
            if cfg.metric_residual_policy=="adaptive":
                converged,restart=assess_residual(R,truth,threshold,check,cfg.metric_gap_ratio,xp=xp)
                active &= ~converged
            else:
                restart=check.copy()
                rn=xp.where(check,norm(truth),rn)
                active &= rn>threshold
            R[:]=xp.where(restart[None,:],truth,R);restarts+=xp.sum(restart)
            Z=m.precondition(xp.where(active[None,:],R,0),out=Zwork)
            newrz=xp.sum(R.conj()*Z,axis=0).real
            beta=xp.where(active & ~restart,newrz/xp.where(active & (rz!=0),rz,1),0)
            P[:]=xp.where(active[None,:],Z+P*beta,0);rz=newrz
            if (k+1)%cfg.metric_gpu_check_batch==0 or k+1==cfg.metric_maxiter:
                # One small transfer per batch. Errors are latched on device.
                status=xp.stack((xp.any(active),xp.any(broken))).get()
                m.control_status_reads+=1;m.control_status_bytes+=status.nbytes
                if not status[0] or status[1]:break
        m.true_residual_checks+=int(checks);m.residual_restarts+=int(restarts)
        return Y.copy(),iters,broken
