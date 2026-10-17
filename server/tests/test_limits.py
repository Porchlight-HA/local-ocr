import os

from local_ocr_server.limits import effective_cpus

AVAILABLE = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def test_effective_cpus_quota(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("400000 100000\n")
    assert effective_cpus(str(p)) == 4


def test_effective_cpus_fractional_rounds_up(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("150000 100000\n")
    assert effective_cpus(str(p)) == 2


def test_effective_cpus_tiny_quota_is_at_least_one(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("10000 100000\n")
    assert effective_cpus(str(p)) == 1


def test_effective_cpus_unlimited_uses_affinity(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("max 100000\n")
    assert effective_cpus(str(p)) == AVAILABLE


def test_effective_cpus_missing_file():
    assert effective_cpus("/nonexistent") == AVAILABLE


def test_effective_cpus_garbage_file(tmp_path):
    p = tmp_path / "cpu.max"
    p.write_text("garbage\n")
    assert effective_cpus(str(p)) == AVAILABLE


def test_effective_cpus_falls_back_to_cpu_count_without_affinity(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    assert effective_cpus("/nonexistent") == 7
