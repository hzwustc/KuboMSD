"""Align internally integrated ABACUS harmonics with the declared H basis.

ABACUS ordering is 0, cos(1), sin(1), ... with Condon-Shortley signs.
DeePTB/dftio uses m=-l..l with these signs removed. The signed permutation
is the ABACUS2DFTIO convention; no radial normalization or H modification.
"""
import numpy as np

CONVENTIONS = ('abacus-real', 'deeptb-real')


def shell_order(l, convention):
    if convention == 'abacus-real':
        return list(range(2*l+1)), [1.]*(2*l+1), list(range(2*l+1))
    if convention != 'deeptb-real':
        raise ValueError('unsupported Hamiltonian orbital_convention')
    ms = list(range(-l, l+1))
    source = [0 if m == 0 else (2*m-1 if m > 0 else -2*m) for m in ms]
    return source, [(-1.)**abs(m) for m in ms], ms


def orbital_labels(orbital, convention):
    return [f'l{l}:z{z}:m{m}' for l, z, _ in orbital.radial
            for m in shell_order(l, convention)[2]]


def align_integral_coo(rows, cols, values, parsed, context):
    convention = parsed.harmonic_convention
    if convention not in CONVENTIONS:
        raise ValueError('unsupported Hamiltonian orbital_convention')
    if convention == 'abacus-real':
        return rows, cols, values
    source, signs, offset = [], [], 0
    for symbol in parsed.structure['symbols']:
        for l, z, _ in parsed.orbitals[symbol].radial:
            permutation, phase, _ = shell_order(l, convention)
            source.extend(offset + np.asarray(permutation))
            signs.extend(phase)
            offset += 2*l+1
    inverse = np.argsort(source).astype(np.int64)
    source_signs = np.asarray(signs)[inverse]
    context.require_bytes(40*len(rows)+16*offset, 'orbital basis permutation workspace')
    with context.phase('harmonic_alignment'):
        mapping = context.asarray(inverse, 'orbital_source_to_target', 'constants')
        sign = context.asarray(source_signs, 'orbital_source_signs', 'constants')
        values *= (sign[rows]*sign[cols])[None, :]
        return mapping[rows], mapping[cols], values
