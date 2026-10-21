import pytest
from kubo_msd.config import RunConfig


def test_cpu_stochastic_defaults_to_native_with_bounded_threads():
    cfg = RunConfig([0.], [0.], solver="stochastic-kpm", kernel="jackson")
    assert getattr(cfg, "cpu_kernel", None) == "cpp"
    assert getattr(cfg, "omp_threads", None) == 1
    assert getattr(cfg, "transport_distribution", None) == "serial"


@pytest.mark.parametrize("kwargs", [
    {"cpu_kernel": "bogus"},
    {"omp_threads": 0},
    {"omp_threads": True},
    {"transport_distribution": "bogus"},
    {"transport_distribution": "mpi"},
])
def test_invalid_acceleration_and_exact_mpi_rejected(kwargs):
    with pytest.raises(ValueError):
        RunConfig([0.], [0.], **kwargs)
