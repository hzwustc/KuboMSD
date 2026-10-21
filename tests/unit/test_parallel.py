import importlib.util
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
import pytest


def test_partition_preserves_global_blocks_and_samples():
    assert importlib.util.find_spec("kubo_msd.parallel") is not None
    from kubo_msd.parallel import owned_blocks
    for ns, block_size, world in [(19, 3, 4), (5, 2, 4), (1, 4, 3)]:
        all_ids = []
        for rank in range(world):
            blocks = owned_blocks(ns, block_size, rank, world)
            for block in blocks:
                all_ids.extend(range(block*block_size, min(ns, (block+1)*block_size)))
        assert sorted(all_ids) == list(range(ns))
        assert len(all_ids) == len(set(all_ids))


def test_serial_context_never_imports_mpi():
    p = subprocess.run([sys.executable, "-c", """
import sys
from kubo_msd.parallel import ParallelContext
p = ParallelContext()
assert p.rank == 0 and p.size == 1
assert p.stage('arithmetic', lambda: 13) == 13
assert 'mpi4py' not in sys.modules
"""], capture_output=True, text=True, env={
    **os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]/"src")})
    assert p.returncode == 0, p.stderr


def test_serial_gather_keeps_block_order_and_values():
    from kubo_msd.parallel import ParallelContext
    values = np.arange(24.).reshape(3, 2, 4)
    gathered = ParallelContext().gather_blocks(values, [0, 1, 2], 3)
    np.testing.assert_array_equal(gathered, values)


def test_invalid_partition_parameters_fail():
    from kubo_msd.parallel import owned_blocks
    with pytest.raises(ValueError):
        owned_blocks(2, 0, 0, 1)
    with pytest.raises(ValueError):
        owned_blocks(2, 1, 2, 2)
