from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
import numpy as np


@dataclass(frozen=True)
class GpuBuildOptions:
    integral_tolerance: float = 1e-5
    table_settings: dict = field(default_factory=dict)
    pair_batch_size: int = 128
    device_memory_limit_bytes: int | None = None
    table_cache_dir: Any = None
    progress_path: Any = None

    def __post_init__(self):
        if not np.isfinite(self.integral_tolerance) or self.integral_tolerance <= 0:
            raise ValueError('integral tolerance must be finite positive')
        if type(self.pair_batch_size) is not int or self.pair_batch_size <= 0:
            raise ValueError('pair batch size must be positive integer')
        if self.device_memory_limit_bytes is not None and (
                type(self.device_memory_limit_bytes) is not int or self.device_memory_limit_bytes <= 0):
            raise ValueError('device memory limit must be positive integer')
        object.__setattr__(self, 'table_settings', MappingProxyType(dict(self.table_settings)))


@dataclass(frozen=True)
class ParsedBuildInput:
    structure: dict
    orbitals: dict
    h_images: dict
    orbital_order: tuple
    atom_slices: tuple
    input_hashes: tuple
    source_path: str
    harmonic_convention: str = 'abacus-real'


@dataclass(frozen=True)
class PairPlan:
    keys: np.ndarray
    species: np.ndarray
    displacement_A: np.ndarray
    pair_ids: np.ndarray
    shapes: np.ndarray
    offsets: np.ndarray
    report: dict

    def batch(self, start, stop):
        return PairPlan(self.keys[start:stop], self.species[start:stop],
                        self.displacement_A[start:stop], self.pair_ids[start:stop],
                        self.shapes[start:stop], self.offsets[start:stop+1]-self.offsets[start], self.report)


@dataclass(frozen=True)
class DeviceTables:
    arrays: dict
    metadata: dict
    identity: str
    device_id: int


@dataclass(frozen=True)
class DevicePairBlocks:
    overlap: Any
    dipole: Any
    pair_ids: Any
    offsets: Any
    batch: PairPlan
    device_id: int


@dataclass(frozen=True)
class DeviceModelBundle:
    H_cov: Any
    S: Any
    device_id: int
    velocity: dict = field(default_factory=dict)
    position: dict = field(default_factory=dict)
    composite: dict = field(default_factory=dict)
    orbital_order: tuple = ()
    atom_slices: tuple = ()
    provenance: dict = field(default_factory=dict)
    basis: str = 'nonorthogonal'
    cache_identity: str = ''

    def __post_init__(self):
        matrices = [self.H_cov, self.S, *self.velocity.values(), *self.position.values()]
        matrices += [x for parts in self.composite.values() for x in parts]
        for a in matrices:
            if a.shape != self.H_cov.shape or a.dtype != np.complex128:
                raise ValueError('device operators require identical square shapes and complex128')
            if a.shape[0] != a.shape[1] or any(x.device.id != self.device_id for x in (a.data,a.indices,a.indptr)):
                raise ValueError('invalid shape or wrong device')
        for name in ('velocity','position','composite','provenance'):
            object.__setattr__(self,name,MappingProxyType(dict(getattr(self,name))))

    @property
    def n_orbitals(self): return self.H_cov.shape[0]

    @property
    def nnz(self): return self.H_cov.nnz


@dataclass(frozen=True)
class GpuBuildResult:
    model: DeviceModelBundle
    report: dict


# The context implementation is lazy with respect to CuPy.
from ...backends.gpu_context import GPUContext
