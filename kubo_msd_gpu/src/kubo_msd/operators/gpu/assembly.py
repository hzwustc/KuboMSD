import numpy as np


def stable_csr(rows,cols,sequence,values,n,context):
    xp=context.xp;count=len(rows)
    if n>=2**31 or count>=2**31:raise ValueError('CSR exceeds int32 library indexing')
    context.require_bytes(count*8*7+n*8+count*16,'stable CSR sorting/merge')
    with context.phase('csr_assembly'):
        if count==0:return [context.sparse.csr_matrix((n,n),dtype=xp.complex128) for _ in range(len(values))]
        order=xp.lexsort(xp.stack((sequence,cols,rows))).astype(xp.int64)
        rr=rows[order];cc=cols[order]
        starts=xp.flatnonzero(xp.r_[True,(rr[1:]!=rr[:-1])|(cc[1:]!=cc[:-1])]).astype(xp.int64)
        row=rr[starts];col=cc[starts].astype(xp.int32)
        ptr=xp.r_[xp.int32(0),xp.cumsum(xp.bincount(row,minlength=n))].astype(xp.int32)
        mats=[];kernel=context.kernel('assembly.cu','reduce_segments')
        for v in values:
            out=xp.empty(len(starts),dtype=xp.complex128)
            kernel(((len(starts)+127)//128,),(128,),(v,order,starts,np.int32(len(starts)),np.int64(count),out))
            # Preserve explicit zeros: exact cache round trips and deterministic support.
            mats.append(context.sparse.csr_matrix((out,col,ptr),shape=(n,n)))
        return mats


def pair_coo(blocks,parsed,context):
    xp=context.xp;b=blocks.batch;total=len(blocks.overlap);npair=len(b.pair_ids)
    context.require_bytes(total*2*(8*3+16*7),'S/dipole COO fill')
    with context.phase('csr_assembly'):
        keys=context.asarray(b.keys,'pair_keys','input_upload')
        ao=context.asarray(np.array([a for a,b in parsed.atom_slices]+[parsed.atom_slices[-1][1]],np.int64),'atom_offsets')
        d=context.asarray(b.displacement_A,'pair_displacement','input_upload')
        rows=xp.empty(2*total,dtype=xp.int64);cols=xp.empty_like(rows);seq=xp.empty_like(rows);val=xp.empty((7,2*total),dtype=xp.complex128)
        if total:context.kernel('assembly.cu','fill_pair_coo')(((2*total+127)//128,),(128,),
            (keys,blocks.offsets,ao,blocks.pair_ids,d,blocks.overlap,context.contiguous(blocks.dipole,'pair_D'),np.int32(npair),np.int64(total),rows,cols,seq,val))
        return rows,cols,seq,val


def h_coo(parsed,context):
    xp=context.xp;images={}
    # Upload original H(R) blocks once. Validate supplied partners on device,
    # and generate missing Hermitian images on device without altering H.
    for key,v in sorted(parsed.h_images.items()):images[key]=context.asarray(v,'H_image','input_upload')
    for key,v in list(images.items()):
        i,j,*R=key;partner=(j,i,*(-np.array(R)))
        if partner in images:
            if not bool(xp.allclose(images[partner],v.conj().T,rtol=1e-10,atol=1e-12)):
                raise ValueError('inconsistent R/-R Hamiltonian blocks')
        else:
            images[partner]=context.contiguous(v.conj().T,'H_mirror')
    keys=sorted(images);off=np.r_[0,np.cumsum([images[k].size for k in keys])].astype(np.int64)
    total=int(off[-1]);ao=np.array([a for a,b in parsed.atom_slices]+[parsed.atom_slices[-1][1]],np.int64)
    st=parsed.structure;pos=st['positions_A'];cell=st['cell_A']
    d=np.array([pos[k[1]]-pos[k[0]]+np.array(k[2:])@cell for k in keys],float).reshape(-1,3)
    context.require_bytes(total*(16*5+8*3),'H COO fill')
    with context.phase('csr_assembly'):
        data=xp.concatenate([images[k].ravel() for k in keys]) if keys else xp.empty(0,dtype=xp.complex128)
        rows=xp.empty(total,dtype=xp.int64);cols=xp.empty_like(rows);seq=xp.empty_like(rows);val=xp.empty((4,total),dtype=xp.complex128)
        if total:context.kernel('assembly.cu','fill_h_coo')(((total+127)//128,),(128,),
            (context.asarray(np.array(keys,np.int64),'H_keys'),context.asarray(off,'H_offsets'),context.asarray(ao,'atom_offsets'),
             context.asarray(d,'H_displacements'),data,np.int32(len(keys)),np.int64(total),rows,cols,seq,val))
    return rows,cols,seq,val
