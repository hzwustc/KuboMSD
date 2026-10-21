"""Unnormalised random phases and sparse polynomial metric correction."""
import numpy as np
from scipy.fft import dct
import scipy.sparse as sp
from .bounds import spd_bounds
from ..errors import NumericalError

def random_phases(n,seed,sample_ids):
    return np.column_stack([np.exp(2j*np.pi*np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed),int(i)]))).random(n)) for i in sample_ids])

class MetricSampler:
    def __init__(self,metric,config):
        self.metric=metric; self.backend=metric.backend; self.xp=metric.xp; xp=self.xp
        self.z=1/xp.sqrt(metric.diagonal)
        # Initialization bounds on CPU; no repeated host vector transfers.
        if hasattr(metric,'context'):
            from .gpu_setup import spd_bounds_gpu
            W=metric.S.multiply(self.z[:,None]).multiply(self.z[None,:]).tocsr()
            lo,hi,report=spd_bounds_gpu(W,metric.context)
            del W
        else:
            host=metric.S if self.backend.name=='cpu' else metric.S.get()
            z=1/np.sqrt(host.diagonal().real); W=sp.diags(z)@host@sp.diags(z)
            lo,hi,report=spd_bounds(W)
        self.center=(hi+lo)/2; self.half=(hi-lo)/2
        self.coeff=None; self.report={'bounds':[lo,hi],'bounds_report':report,'preconditioner':'diagonal congruence','target':config.sampling_tol}
        if self.half<1e-14*self.center:
            self.coeff=np.array([1/np.sqrt(self.center)]); self.report.update(order=0,max_relative_grid_error=0.); return
        for order in (2**k for k in range(4,20)):
            order=min(order,config.sampling_max_order)
            n=max(2*order,64); theta=np.pi*(np.arange(n)+.5)/n
            f=1/np.sqrt(self.center+self.half*np.cos(theta))
            coef=dct(f,type=2)/n; coef[0]*=.5; coef=coef[:order+1]
            xx=np.cos(np.linspace(0,np.pi,max(4097,8*order+1)))
            error=np.max(np.abs(np.polynomial.chebyshev.chebval(xx,coef)*np.sqrt(self.center+self.half*xx)-1))
            if error<config.sampling_tol*.25:
                self.coeff=coef; self.report.update(order=order,max_relative_grid_error=float(error)); break
            if order>=config.sampling_max_order: break
        if self.coeff is None: raise NumericalError('metric inverse sqrt polynomial failed sampling tolerance')

    def transform(self,xi):
        xp=self.xp; xi=xp.asarray(xi,dtype=xp.complex128); coeff=self.coeff
        if self.metric.native is not None:
            return self.metric.native.sample(xi,coeff,self.center,self.half)
        if len(coeff)==1: return self.z[:,None]*xi*coeff[0]
        def action(B): return (self.z[:,None]*(self.metric.S@(self.z[:,None]*B))-self.center*B)/self.half
        b1=xp.zeros_like(xi); b2=xp.zeros_like(xi)
        for c in coeff[:0:-1]: b0=2*action(b1)-b2+c*xi; b2,b1=b1,b0
        return self.z[:,None]*(action(b1)-b2+coeff[0]*xi)

def sample_metric(seed,sample_ids,ops,config):
    if not hasattr(ops,'sampler'): ops.sampler=MetricSampler(ops.metric,config)
    xi=random_phases(ops.n,seed,sample_ids)
    if hasattr(ops,'context'): xi=ops.context.asarray(xi,'PCG64_phases','random_input')
    return ops.sampler.transform(xi)
