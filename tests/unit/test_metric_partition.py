import json
import numpy as np
import pytest
from kubo_msd.metric.partition import make_spatial_partition, load_metric_partition


def test_periodic_whole_atom_deterministic_partition_and_stale_rejection(tmp_path):
    structure=dict(cell_A=np.diag([10.,10,10]),positions_A=[[.1,0,0],[5,0,0],[9.9,0,0]],
                   symbols=["A","B","B"])
    slices=[(0,2),(2,3),(3,4)]
    args=dict(max_orbitals=3,s_fingerprint="s",physical_key="key",orbital_order_hash="order")
    d=make_spatial_partition(structure,slices,**args)
    assert d["blocks"][0]["atoms"]==[0,2]
    assert d["blocks"][0]["orbitals"]==[0,1,3]
    assert d==make_spatial_partition(structure,slices,**args)
    p=tmp_path/"partition.json";p.write_text(json.dumps(d))
    kwargs=dict(s_fingerprint="s",physical_key="key",orbital_order_hash="order",
                atom_slices=slices,n_orbitals=4)
    assert load_metric_partition(p,**kwargs)==d
    with pytest.raises(ValueError):
        load_metric_partition(p,**{**kwargs,"s_fingerprint":"new"})
    d["blocks"][1]["orbitals"]=[0]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_metric_partition(p,**kwargs)


def test_triclinic_nearest_image_not_fractional_rounding():
    # Near-collinear primitive vectors: nearest image needs a mixed translation.
    cell=np.array([[3.,0,0],[2.9,.3,0],[0,0,6.]])
    frac=np.array([[0,0,0],[.49,.49,0],[.1,0,0]])
    d=make_spatial_partition(dict(cell_A=cell,positions_A=frac@cell,symbols=["A"]*3),
                             [(0,1),(1,2),(2,3)],max_orbitals=2,
                             s_fingerprint="s",physical_key="p",orbital_order_hash="o")
    assert d["blocks"][0]["atoms"]==[0,1]


def test_partition_rejects_oversized_atom():
    with pytest.raises(ValueError):
        make_spatial_partition(dict(cell_A=np.eye(3),positions_A=[[0,0,0]],symbols=["A"]),
                               [(0,4)],max_orbitals=3,s_fingerprint="s",physical_key="p",orbital_order_hash="o")
