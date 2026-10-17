import logging
import os
import socket
import stat
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from local_ocr_server import discovery
from local_ocr_server.discovery import advertise_zeroconf, announce_supervisor, load_or_create_token


def test_token_persisted_and_reused(tmp_path):
    path = str(tmp_path / "token")
    first = load_or_create_token(path)
    assert len(first) >= 43
    assert load_or_create_token(path) == first


def test_token_file_mode_0600(tmp_path):
    path = tmp_path / "sub" / "token"
    load_or_create_token(str(path))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


@pytest.fixture
def sup_env(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_TOKEN", "sup-secret")


@pytest.fixture
async def fake_supervisor(aiohttp_server, monkeypatch):
    """Fake Supervisor; tests set .routes behaviour via the returned state dict."""
    state = {"self_status": 200, "apps_status": 200, "post_status": 200, "seen": []}

    async def info(request):
        key = "self_status" if "/addons/" in request.path else "apps_status"
        state["seen"].append((request.method, request.path, request.headers.get("Authorization")))
        if state[key] != 200:
            return web.Response(status=state[key])
        return web.json_response({"data": {"hostname": "abcd1234-local-ocr"}})

    async def post(request):
        state["seen"].append((request.method, request.path, request.headers.get("Authorization")))
        state["body"] = await request.json()
        return web.json_response({}, status=state["post_status"])

    app = web.Application()
    app.router.add_get("/addons/self/info", info)
    app.router.add_get("/apps/self/info", info)
    app.router.add_post("/discovery", post)
    server = await aiohttp_server(app)
    monkeypatch.setattr(discovery, "SUPERVISOR", f"http://127.0.0.1:{server.port}")
    return state


async def test_announce_posts_expected_body(sup_env, fake_supervisor):
    async with aiohttp.ClientSession() as s:
        assert await announce_supervisor(s, "TOKEN", 8738) is True
    assert fake_supervisor["body"] == {
        "service": "local_ocr",
        "config": {"host": "abcd1234-local-ocr", "port": 8738, "token": "TOKEN"},
    }
    assert all(auth == "Bearer sup-secret" for _, _, auth in fake_supervisor["seen"])


async def test_announce_falls_back_to_apps_on_404(sup_env, fake_supervisor):
    fake_supervisor["self_status"] = 404
    async with aiohttp.ClientSession() as s:
        assert await announce_supervisor(s, "T", 1) is True
    assert [p for _, p, _ in fake_supervisor["seen"]] == ["/addons/self/info", "/apps/self/info", "/discovery"]


async def test_announce_without_supervisor_token_returns_false(monkeypatch):
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    async with aiohttp.ClientSession() as s:
        assert await announce_supervisor(s, "T", 8738) is False


async def test_announce_supervisor_500_logs_warning_returns_false(sup_env, fake_supervisor, caplog):
    fake_supervisor["post_status"] = 500
    async with aiohttp.ClientSession() as s:
        with caplog.at_level(logging.WARNING):
            assert await announce_supervisor(s, "SECRETTOK", 8738) is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "SECRETTOK" not in caplog.text


async def test_announce_unreachable_returns_false(sup_env, monkeypatch):
    monkeypatch.setattr(discovery, "SUPERVISOR", "http://127.0.0.1:1")
    async with aiohttp.ClientSession() as s:
        assert await announce_supervisor(s, "T", 8738) is False


@pytest.mark.parametrize("required,auth", [(True, b"1"), (False, b"0")])
async def test_zeroconf_txt_records(monkeypatch, required, auth):
    az = MagicMock()
    az.async_register_service = AsyncMock()
    monkeypatch.setattr(discovery, "AsyncZeroconf", lambda *a, **k: az)
    monkeypatch.setattr(discovery, "local_ipv4_addresses", lambda: ["192.168.1.5"])
    monkeypatch.setattr(socket, "gethostname", lambda: "myhost")
    assert await advertise_zeroconf(8738, required) is az
    info = az.async_register_service.await_args.args[0]
    assert info.type == "_local-ocr._tcp.local."
    assert info.name == "Local OCR on myhost._local-ocr._tcp.local."
    assert info.port == 8738
    assert info.properties == {b"api": b"1", b"auth": auth}
    assert info.parsed_addresses() == ["192.168.1.5"]


async def test_zeroconf_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("no multicast")
    monkeypatch.setattr(discovery, "AsyncZeroconf", boom)
    monkeypatch.setattr(discovery, "local_ipv4_addresses", lambda: ["192.168.1.5"])
    assert await advertise_zeroconf(8738, False) is None


async def _noop(captured, app):
    captured.update(app=app)


def test_main_modes(monkeypatch, tmp_path):
    from local_ocr_server import __main__ as m

    captured = {}
    monkeypatch.setattr(m, "serve", lambda app, *a, **k: _noop(captured, app))
    monkeypatch.setattr(m, "OcrEngine", lambda **k: object())
    monkeypatch.setattr(m, "create_app", lambda e, t, v: captured.update(token=t) or m.web.Application())

    monkeypatch.setenv("SUPERVISOR_TOKEN", "sup")
    monkeypatch.setenv("LOCAL_OCR_TOKEN", "ignored")
    m.main(["--token-file", str(tmp_path / "t")])
    assert captured["token"] == (tmp_path / "t").read_text().strip() != "ignored"
    assert captured["app"]["standalone"] is False

    m.main(["--standalone"])
    assert captured["token"] == "ignored"
    assert captured["app"]["standalone"] is True


async def test_announce_happens_after_server_is_listening(sup_env, aiohttp_server, monkeypatch):
    import asyncio

    from local_ocr_server import __main__ as m
    from local_ocr_server.app import create_app

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    health = {}

    async def info(request):
        return web.json_response({"data": {"hostname": "h"}})

    async def post(request):
        try:
            async with aiohttp.ClientSession() as s, s.get(f"http://127.0.0.1:{port}/v1/health") as r:
                health["status"] = r.status
        except aiohttp.ClientError as err:
            health["error"] = repr(err)
        return web.json_response({})

    sup = web.Application()
    sup.router.add_get("/addons/self/info", info)
    sup.router.add_post("/discovery", post)
    server = await aiohttp_server(sup)
    monkeypatch.setattr(discovery, "SUPERVISOR", f"http://127.0.0.1:{server.port}")

    app = create_app(MagicMock(close=AsyncMock(), state="idle", last_error=None), "tok", "test")
    after = m.make_after_listen(app, True, "tok", port)
    stop = asyncio.Event()
    run = asyncio.create_task(m.serve(app, "127.0.0.1", port, after, stop))
    for _ in range(100):
        if health:
            break
        await asyncio.sleep(0.05)
    stop.set()
    await run
    assert health == {"status": 200}
