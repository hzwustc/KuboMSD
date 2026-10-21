from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any

class Reason(IntFlag):
    LOW_DOS=1
    UNCERTAIN_TRACE=2
    NO_SELECTED_STATES=4
    TRUNCATION_UNVERIFIED=8
    INTEGRATION_UNVERIFIED=16
    INSUFFICIENT_BLOCKS=32

@dataclass
class ModelBundle:
    H_cov: Any
    S: Any
    velocity: dict=field(default_factory=dict)
    position: dict=field(default_factory=dict)
    composite: dict=field(default_factory=dict)
    orbital_order: list=field(default_factory=list)
    atom_slices: list=field(default_factory=list)
    provenance: dict=field(default_factory=dict)
    basis: str='nonorthogonal'
    @property
    def n_orbitals(self): return self.H_cov.shape[0]

@dataclass
class IntegralBundle:
    overlap_images: dict
    dipole_images: dict
    orbital_order: list
    diagnostics: dict

@dataclass
class MetricSolveReport:
    converged: Any
    iterations: Any
    true_residual: Any
    relative_residual: Any
    breakdown: Any

@dataclass
class TransportBlock:
    start: int
    time_fs: Any
    arrays: dict
    static: dict=field(default_factory=dict)
    metadata: dict=field(default_factory=dict)

@dataclass
class RunSummary:
    output_path: str
    run_status: str
    method: str
    backend: str
    diagnostics: dict=field(default_factory=dict)
