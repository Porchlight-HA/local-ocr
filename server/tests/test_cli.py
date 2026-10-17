import pytest

from local_ocr_server.__main__ import build_parser


def test_log_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert build_parser().parse_args([]).log_level == "info"


def test_log_level_env_and_flag(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert build_parser().parse_args([]).log_level == "warning"
    assert build_parser().parse_args(["--log-level", "debug"]).log_level == "debug"


def test_bad_env_log_level_falls_back(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "verbose")
    assert build_parser().parse_args([]).log_level == "info"


def test_bad_flag_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--log-level", "verbose"])
