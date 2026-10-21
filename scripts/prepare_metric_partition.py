#!/usr/bin/env python3
"""Generate a sidecar only after checking the original structure provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from kubo_msd.io.model import load_model
from kubo_msd.io.legacy_input import read_poscar
from kubo_msd.metric.partition import make_spatial_partition,model_identity
from kubo_msd.validation import matrix_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--structure",required=True,type=Path)
    p.add_argument("--max-orbitals",type=int,choices=[64,128,256],default=128)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    model=load_model(a.model)
    records=model.provenance["construction_provenance"]["files"]
    sha=hashlib.sha256(a.structure.read_bytes()).hexdigest()
    if not any(Path(r["path"]).resolve()==a.structure.resolve() and r["sha256"]==sha for r in records):
        raise ValueError("structure is not the original verified model structure")
    structure=read_poscar(a.structure)
    if len(structure["symbols"])!=len(model.atom_slices):raise ValueError("atom count mismatch")
    # Order is explicitly checked instead of inferred from species grouping.
    for atom,(start,stop) in enumerate(model.atom_slices):
        if not all(str(label).startswith(f"atom{atom}:") for label in model.orbital_order[start:stop]):
            raise ValueError("orbital order/atom mapping mismatch")
    result=make_spatial_partition(structure,model.atom_slices,max_orbitals=a.max_orbitals,
                                 s_fingerprint=matrix_fingerprint(model.S),**model_identity(model))
    result.update(structure_path=str(a.structure.resolve()),structure_sha256=sha)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("x") as f:json.dump(result,f,indent=2)
    print(a.out)


if __name__=="__main__":main()
