"""Resident block sums, optional explicit block scratch, device ratio statistics."""
from dataclasses import replace
from pathlib import Path
import tempfile
import shutil
import numpy as np
from ..contracts import TransportBlock, Reason
from ..errors import NumericalError
from ..metric.gpu_bounds import estimate_spectral_bounds_gpu
from ..metric.sampling import sample_metric
from ..kernels.energy import dos_floor, kernel_resolution
from ..observables.diffusion import msd_gradient
from ..observables.statistics_gpu import estimate_ratio_gpu
from .stepwise import stepwise_samples
from .spectral import spectral_samples, spectral_workspace_bytes, maximum_segment_order


def iter_stochastic_blocks_gpu(ops, config, method, samples=None):
    from .stochastic import _quantity_keys
    ctx,xp=ops.context,ops.xp
    with ctx.activate():
        with ctx.phase('spectral_bounds'):
            report=estimate_spectral_bounds_gpu(ops,config)
        cfg=replace(config,spectral_bounds=report['bounds'])
        ns=cfg.nsamples if samples is None else samples.shape[1]
        counts=np.array([min(cfg.statistics_block,ns-i) for i in range(0,ns,cfg.statistics_block)])
        nb=len(counts); na=len(cfg.axes); nt=len(cfg.time_fs); ne=len(cfg.energy_eV)
        keys=_quantity_keys(cfg,method); shape=(nb,na,nt,ne)
        total=int(np.prod(shape))*8*len(keys)
        # Includes sample output, propagation chains, PCG and projected moment scratch.
        requested_batch=cfg.sample_batch
        batch=min(requested_batch,cfg.statistics_block,ns)
        order=maximum_segment_order(cfg) if method=='spectral' or cfg.include_vac else None
        while True:
            working=8*batch*na*nt*ne*len(keys)*3+16*ops.n*batch*(cfg.n_moments+128+na*16)
            if order is not None:
                working+=spectral_workspace_bytes(ops.n,batch,na,nt,ne,cfg.n_moments,order)
            try:
                # Extra headroom covers persistent sums and simultaneous segments.
                ctx.require_bytes(working*2+nb*ne*16,'stochastic sample workspace',int(cfg.memory_limit_gb*1024**3))
                break
            except NumericalError:
                if batch==1: raise
                batch=max(1,batch//2)
        cfg=replace(cfg,sample_batch=batch)
        active=xp.get_default_memory_pool().used_bytes()
        limit=min(ctx.memory_limit_bytes,int(cfg.memory_limit_gb*1024**3))
        spill=active+2*working+total>limit*.8
        temporary=tempfile.TemporaryDirectory(prefix='kubomsd-gpu-stat-') if spill else None
        if spill and total>shutil.disk_usage(temporary.name).free*.8:
            temporary.cleanup()
            raise NumericalError('RESOURCE_BLOCKED: insufficient statistics scratch disk')
        arrays={}
        try:
            for key in keys:
                if spill:
                    arrays[key]=np.lib.format.open_memmap(Path(temporary.name)/(key.replace('/','_')+'.npy'),
                                                         mode='w+',dtype=float,shape=shape)
                else: arrays[key]=xp.zeros(shape)
            dos_blocks=xp.zeros((nb,ne)); diagnostics=[]
            with ctx.phase('transport'):
                for bi,count in enumerate(counts):
                    sums={key:xp.zeros((na,nt,ne)) for key in keys} if spill else {key:arrays[key][bi] for key in keys}
                    first=bi*cfg.statistics_block
                    for start in range(first,first+count,cfg.sample_batch):
                        end=min(first+count,start+cfg.sample_batch)
                        ids=np.arange(start,end)
                        with ctx.phase('sampling'):
                            B=sample_metric(cfg.seed,ids,ops,cfg) if samples is None else ctx.asarray(samples[:,start:end],'supplied_samples','random_input')
                        if method=='stepwise':
                            dos,nums,diag=stepwise_samples(ops,cfg,B)
                            if cfg.include_vac:
                                _,extra,exdiag=spectral_samples(ops,cfg,B); nums.update(extra)
                                diag['additional_vac_path']=exdiag; diag['additional_cost_recorded']=True
                                del extra
                        else: dos,nums,diag=spectral_samples(ops,cfg,B)
                        primary='commutator' if method=='stepwise' else 'vac_integral'
                        if nt>1:
                            nums['diffusion_A2_per_fs/msd_gradient']=msd_gradient(cfg.time_fs,nums['msd_A2/'+primary],axis=2,xp=xp)
                        if not bool(xp.all(xp.isfinite(dos))): raise NumericalError('nonfinite stochastic DOS')
                        diagnostics.append(diag)
                        # Fixed global sample order, independent of sample batch.
                        for local in range(len(ids)): dos_blocks[bi]+=dos[local]
                        for key,value in nums.items():
                            if not bool(xp.all(xp.isfinite(value))): raise NumericalError('nonfinite stochastic numerator')
                            for local in range(len(ids)): sums[key]+=value[local]
                        del B,dos,nums
                    if spill:
                        for key in keys: arrays[key][bi]=ctx.to_host(sums[key],'complete_statistics_block','statistics_scratch')
                    del sums
            with ctx.phase('statistics'):
                dcounts=ctx.asarray(counts[:,None],'block_counts','constants')
                rho,se,stat=estimate_ratio_gpu(dos_blocks,dcounts,counts,ctx)
                threshold=dos_floor(ops.n,cfg.spectral_bounds[1]-cfg.spectral_bounds[0],cfg)
                valid=xp.isfinite(rho)&(rho>threshold)
                reason=xp.where(valid,0,int(Reason.LOW_DOS)).astype(xp.uint32)
                if se is not None:
                    uncertain=~xp.isfinite(se)|(rho-1.96*se<=0)
                    valid&=~uncertain; reason[uncertain]|=int(Reason.UNCERTAIN_TRACE)
                if nb<8: reason|=int(Reason.INSUFFICIENT_BLOCKS)
                static={'dos_states_per_eV':ctx.to_host(rho,'DOS'),
                        'dos_block_sums':ctx.to_host(dos_blocks,'DOS_block_sums'),
                        'block_counts':counts,'sample_count':np.array(ns),'sample_ids':np.arange(ns)}
                if se is not None: static['stderr_dos_states_per_eV']=ctx.to_host(se,'DOS_stderr')
                hv,hr=ctx.to_host(valid,'valid'),ctx.to_host(reason,'reason')
            metadata={'primary_estimator':primary,'primary_diffusion_estimator':'msd_gradient' if method=='stepwise' else 'vac_integral',
                'resolved_config':cfg.as_dict(),'spectral_report':report,'kernel_resolution':kernel_resolution(cfg),
                'dos_floor':threshold,'statistical_method':stat,
                'sample_generator':'NumPy PCG64 SeedSequence([seed,sample_id]); unit modulus complex phases',
                'sampling':getattr(getattr(ops,'sampler',None),'report',{'source':'caller supplied coefficients'}),
                'sample_batch_diagnostics':diagnostics,'raw_accumulation':'numerator_* are means over nsamples; *_block_sums are sums',
                'resource_batch':{'requested':requested_batch,'effective':batch,'maximum_spectral_order':order,'workspace_allowance_bytes':working*2},
                'scratch_bytes':total if spill else 0,'statistics_location':'GPU slabs; explicit block scratch' if spill else 'GPU resident block sums',
                'parallel':{'distribution':'serial','size':1,'sample_ids':list(range(ns))},
                'science_status':{'NUMERICS_VERIFIED':'UNVERIFIED for this run','STATISTICS_VERIFIED':'NOT RUN','TRUNCATION_VERIFIED':'not applicable'}}
            # Reduce only the legal output time slab if bootstrap workspace is large.
            per_time=8*na*ne*(nb*(len(keys)+2)+4*256)
            available=max(0,limit-xp.get_default_memory_pool().used_bytes())
            slab=min(cfg.time_block,nt,int(available*.7/max(per_time,1)))
            if slab<1: raise NumericalError('RESOURCE_BLOCKED: minimum bootstrap slab cannot fit')
            for start in range(0,nt,slab):
                stop=min(nt,start+slab); out={}
                with ctx.phase('statistics'):
                    for key in keys:
                        raw=ctx.asarray(arrays[key][:,:,start:stop,:],key,'statistics_scratch') if spill else arrays[key][:,:,start:stop,:]
                        val,stderr,_=estimate_ratio_gpu(raw,dos_blocks,counts,ctx)
                        val[...,~valid]=xp.nan
                        out[key]=ctx.to_host(val,key)
                        out['numerator_'+key]=ctx.to_host(raw.sum(0)/ns,'numerator_'+key)
                        out['numerator_block_sums/'+key]=ctx.to_host(raw,'numerator_block_sums/'+key)
                        if stderr is not None:
                            stderr[...,~valid]=xp.nan
                            out['stderr_'+key]=ctx.to_host(stderr,'stderr_'+key)
                    dims=(na,stop-start,ne)
                    for estimator in {k.split('/')[-1] for k in keys}:
                        rc=hr.copy()
                        if estimator=='vac_integral': rc|=int(Reason.INTEGRATION_UNVERIFIED)
                        out['valid/'+estimator]=np.broadcast_to(hv,dims).copy()
                        out['reason_code/'+estimator]=np.broadcast_to(rc,dims).copy()
                yield TransportBlock(start,cfg.time_fs[start:stop],out,static if start==0 else {},metadata if start==0 else {})
        finally:
            if temporary:
                for a in arrays.values():
                    if isinstance(a,np.memmap): a._mmap.close()
                temporary.cleanup()
