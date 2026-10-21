"""Signed orbital permutations at the ABACUS-integral / DeePTB-H5 boundary.

Radial tables use real Condon--Shortley harmonics in 0,cos1,sin1,... order.
DeePTB uses signed-m order -l,...,l and removes the (-1)^m phase.
For each (l,zeta) shell: O_deeptb = U_bra O_abacus U_ket.T.
The input Hamiltonian is already in its declared basis and is never rotated.
"""
import numpy as np
from ..errors import KuboMSDError

CONVENTIONS = ('abacus-real', 'deeptb-real')


def convention_for(structure):
    value = structure.get('orbital_convention', 'abacus-real')
    if value not in CONVENTIONS:
        raise KuboMSDError(f'unsupported orbital harmonic convention: {value}')
    return value


def basis_map(orbitals, convention):
    if convention not in CONVENTIONS:
        raise KuboMSDError(f'unsupported orbital harmonic convention: {convention}')
    indices, signs, offset = [], [], 0
    for l, z, radial in orbitals.radial:
        if convention == 'abacus-real':
            indices.extend(range(offset, offset+2*l+1))
            signs.extend([1]*(2*l+1))
        else:
            for m in range(-l, l+1):
                source = 2*abs(m) if m < 0 else (2*m-1 if m > 0 else 0)
                indices.append(offset+source)
                signs.append(-1 if abs(m) % 2 else 1)
        offset += 2*l+1
    return np.asarray(indices, dtype=np.int64), np.asarray(signs)


def transform_block(value, bra_map, ket_map):
    """Transform the final two orbital dimensions, preserving Cartesian axes."""
    pi, si = bra_map
    pj, sj = ket_map
    return value[..., pi[:, None], pj[None, :]] * si[:, None] * sj[None, :]


def orbital_labels(orbitals, convention):
    if convention == 'abacus-real':
        return orbitals.labels
    basis_map(orbitals, convention)  # validate
    return [f'l{l}:z{z}:m{m}:deeptb' for l,z,r in orbitals.radial for m in range(-l,l+1)]
