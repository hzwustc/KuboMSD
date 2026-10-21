"""Numerical payload adapter; native tables never depend on atom indices."""
import numpy as np
from scipy.interpolate import CubicSpline
from ..errors import KuboMSDError

ALGORITHM='fourier-bessel-real-gaunt-v2-onsite'


def native_module():
    try:
        from .. import _operators_native
    except ImportError as exc:
        raise KuboMSDError('C++ operator extension unavailable; rebuild kubo-msd with pybind11/OpenMP') from exc
    if not _operators_native.capabilities()['fast_math_available']:
        raise KuboMSDError('C++ fast operator mathematics unavailable')
    return _operators_native


def orbital_payload(orbitals):
    ls=[]; species=[]; radii=[]; coeff=[]; offsets=[0]; coff=[0]
    for si,orb in enumerate(orbitals.values()):
        for l,z,radial in orb.radial:
            ls.append(l);species.append(si);radii.extend(orb.radius_bohr)
            coeff.extend(CubicSpline(orb.radius_bohr,radial).c.T.ravel())
            offsets.append(len(radii));coff.append(len(coeff))
    return dict(radial_l=np.array(ls,np.int64),radial_species=np.array(species,np.int64),
                radius=np.array(radii,float),coefficients=np.array(coeff,float),
                grid_offsets=np.array(offsets,np.int64),coeff_offsets=np.array(coff,np.int64),
                species_count=len(orbitals))


def resolved_settings(tolerance,settings=None,memory_limit_bytes=2**31):
    if not np.isfinite(tolerance) or tolerance<=0: raise KuboMSDError('tolerance must be positive')
    scale=min(1.,(tolerance/1e-5)**.2)
    result=dict(kmax=32./scale,dk=.1*scale,distance_step=.05*scale,radial_order=4,memory_limit_bytes=memory_limit_bytes)
    unknown=set(settings or {})-set(result)
    if unknown: raise KuboMSDError(f'unknown table settings: {sorted(unknown)}')
    result.update(settings or {})
    return result


def prepare_tables(orbitals,tolerance=1e-5,settings=None,memory_limit_bytes=2**31):
    return native_module().build_tables(orbital_payload(orbitals),resolved_settings(tolerance,settings,memory_limit_bytes))


def validation_pairs(orbitals,step):
    names=list(orbitals);species=[];displacements=[];scales=[]
    directions=np.array([[1.,2.,3.],[-2.,1.,.7],[.4,-1.,2.],[1.,0.,0.],[0.,1.,1.]])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    from ..units import BOHR_A
    for a,oa in enumerate(orbitals.values()):
        for b,ob in enumerate(orbitals.values()):
            cutoff=oa.cutoff_A+ob.cutoff_A
            for j,f in enumerate([0.,.031,.13,.29,.47,.67,.83,.96,.995]):
                # Explicitly withhold interpolation midpoints (except onsite).
                r=0. if f==0 else min(cutoff*(1-1e-8),(np.floor(cutoff*f/(step*BOHR_A))+.5)*step*BOHR_A)
                species.append([a,b]);displacements.append(r*directions[j%len(directions)]);scales.append(max(1.,cutoff))
    return dict(species=np.asarray(species,np.int64),displacement_A=np.asarray(displacements,float),pair_ids=np.arange(len(species),dtype=np.int64)),np.asarray(scales)


def pair_difference(a,b,scales):
    worst=0.;worst_id=None
    for i,scale in enumerate(scales):
        sl=slice(a['offsets'][i],a['offsets'][i+1])
        v=max(np.max(abs(a['S'][sl]-b['S'][sl]),initial=0),np.max(abs(a['D'][:,sl]-b['D'][:,sl]),initial=0)/scale)
        if v>worst:worst=float(v);worst_id=i
    return dict(max_scaled_difference=worst,worst_pair_id=worst_id)


def validated_tables(orbitals,tolerance=1e-5,settings=None,memory_limit_bytes=2**31):
    native=native_module();payload=orbital_payload(orbitals)
    cfg=resolved_settings(tolerance,settings,memory_limit_bytes);history=[]
    for attempt in range(3):
        table=native.build_tables(payload,cfg)
        pairs,scales=validation_pairs(orbitals,table.settings['distance_step'])
        base=native.evaluate_pairs(table,pairs,1)
        transform_cfg={**cfg,'kmax':cfg['kmax']*1.5,'dk':cfg['dk']/2,'radial_order':min(16,cfg['radial_order']+2)}
        refined=native.build_tables(payload,transform_cfg)
        transform=pair_difference(base,native.evaluate_pairs(refined,pairs,1),scales)
        del refined
        interp=native.build_tables(payload,{**cfg,'distance_step':cfg['distance_step']/2})
        interpolation=pair_difference(base,native.evaluate_pairs(interp,pairs,1),scales)
        del interp
        history.append(dict(settings=dict(table.settings),transform=transform,interpolation=interpolation))
        if max(transform['max_scaled_difference'],interpolation['max_scaled_difference'])<=tolerance/5:
            return table,dict(algorithm=ALGORITHM,tolerance=tolerance,budget=tolerance/5,status='PASS',cache='cold',
                              validation_kind='finite validation geometry set; not uniform error bound',
                              reference_accuracy='UNVERIFIED',sample_count=len(scales),history=history,
                              resolved_settings=dict(table.settings),channel_count=table.channel_count)
        cfg={**transform_cfg,'distance_step':cfg['distance_step']/2}
    raise KuboMSDError(f'fast table convergence failed; epsilon/5={tolerance/5}: {history}')
