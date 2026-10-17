"""Resource limits derived from the container's cgroup."""
from __future__ import annotations

import math
import os


def _available_cpus() -> int:
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        return len(getaffinity(0))
    return os.cpu_count() or 1


def effective_cpus(cpu_max_path: str = "/sys/fs/cgroup/cpu.max") -> int:
    """CPUs this process may use: the cgroup v2 quota rounded up, else the affinity mask."""
    try:
        with open(cpu_max_path) as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return _available_cpus()
