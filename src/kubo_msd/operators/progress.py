"""Wall-clock build accounting. Nested spans are reported, never summed."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import resource
import platform
import threading
import time

HEARTBEAT_SECONDS = 15.


def environment_info():
    import numpy, scipy, h5py
    from threadpoolctl import threadpool_info
    from .. import __version__
    try:
        from .. import _operators_native
        compiler = _operators_native.capabilities()
    except ImportError:
        compiler = {"available": False}
    return dict(host=platform.node(),platform=platform.platform(),
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                software=dict(kubo_msd=__version__,python=platform.python_version(),
                              numpy=numpy.__version__,scipy=scipy.__version__,
                              h5py=h5py.__version__),
                native=compiler,blas_pools=threadpool_info())


STAGES = (
    "read_inputs", "neighbor_list", "table_build_or_load", "pair_evaluation",
    "csr_assembly", "matrix_validation", "cache_write", "cache_readback",
    "total_build", "mpi_distribution", "table_broadcast", "rank_compute",
    "shard_write", "merge", "communication_wait",
)


@dataclass
class BuildReport:
    builder: str
    progress_path: object = None
    timings: dict = field(default_factory=lambda: {k: 0.0 for k in STAGES})
    counts: dict = field(default_factory=dict)
    status: str = "RUNNING"
    phase: str = "initialization"
    error: str = ""
    details: dict = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    _write_lock: object = field(default_factory=threading.RLock,repr=False)

    def __post_init__(self):
        self.details["environment"] = environment_info()

    def snapshot(self):
        return dict(schema="kubomsd-build-report-v1", builder=self.builder,
                    status=self.status, phase=self.phase, error=self.error,
                    timings=self.timings, counts=self.counts, details=self.details,
                    elapsed_s=time.perf_counter()-self.started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    timing_policy="wall seconds; nested spans must not be added")

    def update(self, *, heartbeat=False, **counts):
        self.counts.update(counts)
        if self.progress_path:
            with self._write_lock:
                p = Path(self.progress_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name+f".{os.getpid()}.tmp")
                state = self.snapshot()
                if heartbeat:
                    # Detail payloads can be updated by the computing thread.
                    # A heartbeat only promises phase, elapsed time and counts.
                    state = {**state, "details": {}, "heartbeat": True,
                             "timings": dict(self.timings), "counts": dict(self.counts)}
                tmp.write_text(json.dumps(state, indent=2, allow_nan=False))
                os.replace(tmp, p)

    @contextmanager
    def stage(self, name):
        self.phase = name
        start = time.perf_counter()
        self.update()
        stop = threading.Event()
        errors = []
        def pulse():
            while not stop.wait(HEARTBEAT_SECONDS):
                try:
                    self.update(heartbeat=True)
                except Exception as exc:
                    errors.append(exc)
                    return
        worker = threading.Thread(target=pulse,daemon=True) if self.progress_path else None
        if worker: worker.start()
        try:
            yield
            if errors: raise errors[0]
        except BaseException as exc:
            self.status, self.error = "FAIL", str(exc)
            raise
        finally:
            stop.set()
            if worker: worker.join()
            self.timings[name] = self.timings.get(name, 0.0)+time.perf_counter()-start
            self.update()

    def finish(self):
        self.timings["total_build"] = time.perf_counter()-self.started
        self.status = "PASS"
        self.update()
