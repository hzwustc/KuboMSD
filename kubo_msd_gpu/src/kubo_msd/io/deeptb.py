from pathlib import Path
import re
import h5py
import numpy as np
from ..operators.assembly import complete_images
from ..errors import KuboMSDError

def read_deeptb(path,counts,index_base=0):
    path=Path(path)
    if not path.is_file(): raise KuboMSDError(f'missing Hamiltonian file: {path}')
    images={}
    with h5py.File(path,'r') as f:
        if '0' not in f or not isinstance(f['0'],h5py.Group): raise KuboMSDError('DeePTB requires group 0')
        for key,dataset in f['0'].items():
            if not re.fullmatch(r'-?\d+_-?\d+_-?\d+_-?\d+_-?\d+',key): raise KuboMSDError(f'unknown DeePTB block key {key}')
            i,j,*R=map(int,key.split('_')); a=np.asarray(dataset)
            if a.ndim==3 and a.shape[-1]==2: a=a[...,0]+1j*a[...,1]
            images[(i-index_base,j-index_base,*R)]=a
    return complete_images(images,counts)
