"""Optional MPI coordination; no MPI import for serial transport.

Only the Python main thread communicates. C++ OpenMP workers never call MPI.
Numerators are gathered as float64 blocks, before any ratio normalization.
"""
import os
import numpy as np
from .errors import KuboMSDError, NumericalError


def owned_blocks(nsamples, block_size, rank, size):
    if nsamples < 1 or block_size < 1 or size < 1 or not 0 <= rank < size:
        raise KuboMSDError("invalid sample/block/rank partition")
    return list(range(rank, (nsamples + block_size - 1)//block_size, size))


def check_serial_launcher():
    """Prevent accidental multiple writers when mpiexec was used without MPI mode."""
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        value = os.environ.get(key)
        if value and value.isdigit() and int(value) > 1:
            raise KuboMSDError(
                "multiple MPI processes detected; use --transport-distribution mpi "
                "with stochastic-kpm, or launch the serial solver as one process")
    # SLURM_NTASKS is an allocation size, not proof of an srun process launch.
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) > 1:
        raise KuboMSDError("multiple srun tasks detected; select MPI transport or use srun -n 1")


class ParallelContext:
    def __init__(self, comm=None, mpi=None):
        self.comm, self.mpi = comm, mpi
        self.rank = 0 if comm is None else comm.Get_rank()
        self.size = 1 if comm is None else comm.Get_size()

    @property
    def root(self):
        return self.rank == 0

    def allgather(self, value):
        return [value] if self.comm is None else self.comm.allgather(value)

    def broadcast(self, value):
        return value if self.comm is None else self.comm.bcast(value, root=0)

    def stage(self, label, action, root_only=False):
        """Run non-collective work, then agree on success before the next collective."""
        result, error = None, None
        try:
            if not root_only or self.root:
                result = action()
        except Exception as exc:
            if self.comm is None:
                raise
            error = {"rank": self.rank, "type": type(exc).__name__, "message": str(exc)}
        errors = [e for e in self.allgather(error) if e is not None]
        if errors:
            failure = NumericalError(f"MPI stage {label!r} failed: " +
                                     "; ".join(f"rank {e['rank']}: {e['message']}" for e in errors))
            failure.diagnostics = {"mpi_stage": label, "failures": errors}
            raise failure
        return result

    def root_call(self, label, action):
        result = self.stage(label, action, root_only=True)
        return self.broadcast(result)

    def gather_blocks(self, local, block_ids, nblocks):
        """Gather disjoint blocks to root in global order (bounded by a time slab)."""
        if self.comm is None:
            if list(block_ids) != list(range(nblocks)):
                raise NumericalError("serial block coverage mismatch")
            return np.asarray(local)
        local = self.stage("pack statistical slab",
                           lambda: np.ascontiguousarray(local, dtype=np.float64))
        declarations = self.allgather((list(block_ids), local.shape))

        def validate():
            ids = [j for part, _ in declarations for j in part]
            if sorted(ids) != list(range(nblocks)):
                raise NumericalError("MPI statistical blocks must cover the global set exactly once")
            tails = {shape[1:] for _, shape in declarations}
            if len(tails) != 1 or any(shape[0] != len(part) for part, shape in declarations):
                raise NumericalError("MPI statistical block shapes disagree")
            counts = [int(np.prod(shape)) for _, shape in declarations]
            if sum(counts) > np.iinfo(np.int32).max:
                raise NumericalError("MPI slab exceeds portable count limit; reduce time_block")
            return ids, counts

        ids, counts = self.stage("validate statistical slab", validate)
        displacements = np.cumsum([0] + counts[:-1]).tolist()
        packed = self.stage("allocate MPI receive slab",
                            lambda: np.empty((nblocks,) + local.shape[1:], dtype=np.float64),
                            root_only=True)
        receive = [packed, counts, displacements, self.mpi.DOUBLE] if self.root else None
        self.comm.Gatherv(local, receive, root=0)

        def reorder():
            result = np.empty_like(packed)
            result[np.asarray(ids, dtype=int)] = packed
            return result
        return self.stage("order global statistical slab", reorder, root_only=True)

    def metadata(self):
        return {
            "distribution": "mpi" if self.comm is not None else "serial",
            "world_size": self.size,
            "partition": "complete global statistics blocks, round robin",
            "operator_storage": "replicated per rank",
            "reduction": "raw numerator and DOS block sums, normalized once on root",
            "mpi_library": None if self.mpi is None else self.mpi.Get_library_version().strip(),
            "thread_level": None if self.mpi is None else self.mpi.Query_thread(),
        }

    def bind_gpu(self):
        """Select one GPU per node-local rank; refuse overlapping device ownership."""
        if self.comm is None:
            return
        node = self.comm.Split_type(self.mpi.COMM_TYPE_SHARED)
        local_rank, local_size = node.Get_rank(), node.Get_size()
        try:
            def probe():
                import cupy as cp
                count = cp.cuda.runtime.getDeviceCount()
                index = 0 if count == 1 else local_rank
                if index >= count:
                    raise KuboMSDError("not enough visible GPUs for node-local MPI ranks")
                props = cp.cuda.runtime.getDeviceProperties(index)
                uuid = props.get("uuid")
                if uuid is not None:
                    uuid = bytes(uuid).hex()
                return index, uuid or cp.cuda.Device(index).pci_bus_id
            index, info = self.stage("probe MPI GPU assignment", probe)
            assigned = node.allgather(info)
            def activate():
                import cupy as cp
                if len(set(assigned)) != local_size:
                    raise KuboMSDError("MPI GPU transport requires a distinct GPU per local rank; "
                                       "set scheduler GPU binding/CUDA_VISIBLE_DEVICES accordingly")
                cp.cuda.Device(index).use()
            self.stage("activate distinct MPI GPUs", activate)
        finally:
            node.Free()


def context_for(config):
    if config.transport_distribution == "serial":
        check_serial_launcher()
        return ParallelContext()
    try:
        import mpi4py
        # Respect a caller-initialized MPI, otherwise request only FUNNELED.
        mpi4py.rc.thread_level = "funneled"
        from mpi4py import MPI
    except ImportError as exc:
        raise KuboMSDError("MPI transport requires mpi4py built for the loaded MPI library") from exc
    if MPI.Query_thread() < MPI.THREAD_FUNNELED:
        raise KuboMSDError("MPI transport requires MPI_THREAD_FUNNELED or higher")
    ctx = ParallelContext(MPI.COMM_WORLD, MPI)
    configs = ctx.allgather(config.as_dict())
    if any(c != configs[0] for c in configs):
        raise KuboMSDError("MPI ranks must use identical global RunConfig")
    return ctx
