"""Wall-clock build accounting. Nested spans are reported, never summed."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import resource
import time

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

    def snapshot(self):
        return dict(schema="kubomsd-build-report-v1", builder=self.builder,
                    status=self.status, phase=self.phase, error=self.error,
                    timings=self.timings, counts=self.counts, details=self.details,
                    elapsed_s=time.perf_counter()-self.started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    timing_policy="wall seconds; nested spans must not be added")

    def update(self, **counts):
        self.counts.update(counts)
        if self.progress_path:
            p = Path(self.progress_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name+f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=2, allow_nan=False))
            os.replace(tmp, p)

    @contextmanager
    def stage(self, name):
        self.phase = name
        start = time.perf_counter()
        self.update()
        try:
            yield
        except BaseException as exc:
            self.status, self.error = "FAIL", str(exc)
            raise
        finally:
            self.timings[name] = self.timings.get(name, 0.0)+time.perf_counter()-start
            self.update()

    def finish(self):
        self.timings["total_build"] = time.perf_counter()-self.started
        self.status = "PASS"
        self.update()
