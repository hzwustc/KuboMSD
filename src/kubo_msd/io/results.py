"""Versioned, contiguous HDF5 time-block persistence and explicit completion."""
from dataclasses import asdict,is_dataclass
import json
import os
from pathlib import Path
import numpy as np
import h5py
from ..errors import KuboMSDError
from ..contracts import Reason

def json_safe(v):
    if is_dataclass(v): return json_safe(asdict(v))
    if isinstance(v,dict): return {str(k):json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)): return [json_safe(x) for x in v]
    if isinstance(v,np.generic): return json_safe(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    if isinstance(v,Path): return str(v)
    return v

class ResultsWriter:
    def __init__(self,path,config):
        self.path=Path(path).resolve(); self.path.mkdir(parents=True,exist_ok=False); self.cfg=config; self.next=0; self.keys=None; self.done=False
        self.file=h5py.File(self.path/'results.partial.h5','w'); self.file.attrs['schema_version']='1.0'; self.file.attrs['run_status']='partial'
        self.metadata={'schema_version':'1.0','run_status':'partial','resolved_config':config.as_dict(),'axis_order':'axis,time,energy; statistical sums: block,axis,time,energy','units':{'energy':'eV','time':'fs','msd':'angstrom^2','diffusion':'angstrom^2/fs','vac':'angstrom^2/fs^2','dos':'states/eV/system'},'spin_degeneracy_multiplier':1,'reason_codes':{e.name:int(e) for e in Reason},'hbar_eV_fs':0.6582119569509067}
        for key,v in [('energy_eV',config.energy_eV),('time_fs',config.time_fs),('axes_vectors',config.axes_vectors)]: self.file.create_dataset(key,data=v)
        self._save_meta()
    def _save_meta(self):
        temp=self.path/'metadata.json.tmp'; temp.write_text(json.dumps(json_safe(self.metadata),ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(temp,self.path/'metadata.json')
    def __enter__(self): return self
    def __exit__(self,typ,value,tb):
        if not self.done:
            self.metadata['run_status']='failed' if typ else 'partial'; self.metadata['written_time_points']=self.next
            if typ:
                self.metadata['failure']={'type':typ.__name__,'message':str(value)}
                if hasattr(value,'diagnostics'): self.metadata['failure']['numerics']=value.diagnostics
            self.file.attrs['run_status']=self.metadata['run_status']; self.file.flush(); self._save_meta()
        self.file.close()
    def append(self,block):
        if self.done: raise KuboMSDError('cannot append finalized results')
        t=np.asarray(block.time_fs); end=block.start+len(t)
        if block.start!=self.next or end>len(self.cfg.time_fs) or not np.array_equal(t,self.cfg.time_fs[block.start:end]): raise KuboMSDError('TransportBlock must contain contiguous configured times')
        if not len(t) or not block.arrays: raise KuboMSDError('empty TransportBlock')
        keys=set(block.arrays)
        if self.keys is not None and keys!=self.keys: raise KuboMSDError('estimator schema changed during append')
        for key,value in block.arrays.items():
            a=np.asarray(value); axis=2 if key.startswith('numerator_block_sums/') else 1
            expected_ndim=4 if axis==2 else 3
            if a.ndim!=expected_ndim or a.shape[axis]!=len(t) or a.shape[-1]!=len(self.cfg.energy_eV) or a.shape[axis-1]!=len(self.cfg.axes): raise KuboMSDError(f'invalid shape for {key}: {a.shape}')
            if key not in self.file:
                shape=list(a.shape); shape[axis]=len(self.cfg.time_fs); chunks=list(a.shape); chunks[axis]=min(self.cfg.time_block,len(self.cfg.time_fs)); chunks[-1]=min(chunks[-1],256)
                ds=self.file.create_dataset(key,shape=shape,dtype=a.dtype,chunks=tuple(chunks),fillvalue=np.nan if a.dtype.kind=='f' else 0)
                ds.attrs['time_axis']=axis
            index=[slice(None)]*a.ndim; index[axis]=slice(block.start,end); self.file[key][tuple(index)]=a
        if self.next and (block.static or block.metadata): raise KuboMSDError('static fields/metadata are allowed only in first block')
        for key,value in block.static.items():
            if key in self.file: raise KuboMSDError('duplicate static field '+key)
            self.file.create_dataset(key,data=value)
        self.metadata.update(block.metadata); self.keys=keys; self.next=end; self.file.flush(); self.metadata['written_time_points']=self.next; self._save_meta()
    def _derive_gradient(self):
        primary=self.metadata['primary_estimator']; key='diffusion_A2_per_fs/msd_gradient'; nt=len(self.cfg.time_fs)
        if nt<2 or key in self.file: return
        source=self.file['msd_A2/'+primary]; shape=source.shape
        out=self.file.create_dataset(key,shape=shape,dtype=float,chunks=source.chunks); numsrc=self.file.get('numerator_msd_A2/'+primary)
        outnum=self.file.create_dataset('numerator_diffusion_A2_per_fs/msd_gradient',shape=shape,dtype=float,chunks=source.chunks) if numsrc is not None else None
        for first in range(0,nt,self.cfg.time_block):
            end=min(first+self.cfg.time_block,nt); lo=max(0,first-1); hi=min(nt,end+1)
            if lo==0: hi=max(hi,min(nt,3))
            if hi==nt: lo=min(lo,max(0,nt-3))
            edge=2 if hi-lo>2 else 1
            out[:,first:end]=.5*np.gradient(source[:,lo:hi],self.cfg.time_fs[lo:hi],axis=1,edge_order=edge)[:,first-lo:end-lo]
            if outnum is not None: outnum[:,first:end]=.5*np.gradient(numsrc[:,lo:hi],self.cfg.time_fs[lo:hi],axis=1,edge_order=edge)[:,first-lo:end-lo]
        # Separate datasets: HDF5 visititems visits a hard-linked object only once,
        # which would drop an estimator's named mask from NPZ/read_results.
        for prefix in ('valid/','reason_code/'):
            if prefix+primary in self.file:
                src=self.file[prefix+primary]; dst=self.file.create_dataset(prefix+'msd_gradient',shape=src.shape,dtype=src.dtype,chunks=src.chunks)
                for first in range(0,nt,self.cfg.time_block): dst[:,first:first+self.cfg.time_block]=src[:,first:first+self.cfg.time_block]
        self.metadata['msd_gradient']={'source':primary,'edge_order':2 if nt>2 else 1,'time_array':'actual output time_fs'}
    def _aliases(self):
        primary=self.metadata['primary_estimator']; dp=self.metadata.get('primary_diffusion_estimator',primary)
        aliases={}
        for quantity,unit,est in [('msd','A2',primary),('diffusion','A2_per_fs',dp)]:
            key=quantity+'_'+unit+'/'+est
            if key not in self.file: continue
            data=self.file[key]
            for i,axis in enumerate(self.cfg.axes):
                if axis in ('x','y','z'):
                    alias=f'{quantity}_{axis}_{unit}'; ds=self.file.create_dataset(alias,shape=data.shape[1:],dtype=float,chunks=(min(self.cfg.time_block,data.shape[1]),min(256,data.shape[2])))
                    for start in range(0,data.shape[1],self.cfg.time_block): ds[start:start+self.cfg.time_block]=data[i,start:start+self.cfg.time_block]
                    aliases[alias]=key
            avg=self.file.create_dataset(f'{quantity}_direction_average_{unit}',shape=data.shape[1:],dtype=float)
            for start in range(0,data.shape[1],self.cfg.time_block): avg[start:start+self.cfg.time_block]=np.mean(data[:,start:start+self.cfg.time_block],axis=0)
        self.metadata['legacy_aliases']=aliases; self.metadata['direction_average']='arithmetic mean over declared orthonormal directions; no volume normalization'
    def finalize(self,summary):
        if self.done or self.next!=len(self.cfg.time_fs) or summary.run_status!='complete': raise KuboMSDError('cannot mark incomplete computation complete')
        if 'dos_states_per_eV' not in self.file or 'primary_estimator' not in self.metadata: raise KuboMSDError('required result metadata missing')
        self._derive_gradient(); self._aliases(); self.metadata['summary']=json_safe(summary)
        # Export before the commit marker; a failed export cannot leave a complete run.
        if self.cfg.output_npz:
            arrays={}
            total=0
            def gather(name,obj):
                nonlocal total
                if isinstance(obj,h5py.Dataset): total+=obj.size*obj.dtype.itemsize
            self.file.visititems(gather)
            if total>self.cfg.memory_limit_gb*1024**3*.5: raise NumericalError('NPZ export exceeds half the RAM budget; use HDF5 without --npz')
            self.file.visititems(lambda name,obj:arrays.update({name:obj[()]}) if isinstance(obj,h5py.Dataset) else None)
            arrays['schema_version']=np.array('1.0'); arrays['run_status']=np.array('complete'); arrays['primary_estimator']=np.array(self.metadata['primary_estimator'])
            arrays['metadata_json']=np.array(json.dumps(json_safe({**self.metadata,'run_status':'complete'}),allow_nan=False))
            np.savez_compressed(self.path/'results.npz',**arrays)
        self.metadata['run_status']='complete'; self.file.attrs['run_status']='complete'; self.file.attrs['metadata_json']=json.dumps(json_safe(self.metadata),allow_nan=False); self.file.flush()
        os.replace(self.path/'results.partial.h5',self.path/'results.h5'); self._save_meta(); self.done=True

def read_results(path,allow_partial=False):
    p=Path(path)
    if p.is_dir():
        metadata=json.loads((p/'metadata.json').read_text()); p=p/('results.h5' if (p/'results.h5').exists() else 'results.partial.h5')
    else: metadata=None
    arrays={}
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as f:
            if 'schema_version' not in f or str(f['schema_version'])!='1.0' or 'metadata_json' not in f: raise KuboMSDError('unversioned legacy NPZ cannot establish full spectrum/kernel metadata')
            metadata=json.loads(str(f['metadata_json'])); arrays={k:np.array(f[k]) for k in f.files if k!='metadata_json'}
    else:
        with h5py.File(p,'r') as f:
            if f.attrs.get('schema_version')!='1.0': raise KuboMSDError('unknown result schema')
            if metadata is None: metadata=json.loads(f.attrs.get('metadata_json','{}')); metadata['run_status']=f.attrs['run_status']
            f.visititems(lambda name,obj:arrays.update({name:obj[()]}) if isinstance(obj,h5py.Dataset) else None)
    if metadata.get('run_status')!='complete' and not allow_partial: raise KuboMSDError('result is failed/partial; use allow_partial explicitly')
    return {'metadata':metadata,'arrays':arrays}

from ..errors import NumericalError
