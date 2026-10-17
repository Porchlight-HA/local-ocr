"""Token persistence, Supervisor discovery (managed mode) and zeroconf (standalone mode)."""
from __future__ import annotations

import logging
import os
import secrets
import socket
from pathlib import Path

import aiohttp
from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

_LOGGER = logging.getLogger(__name__)

SUPERVISOR = "http://supervisor"
SERVICE_TYPE = "_local-ocr._tcp.local."
_TIMEOUT = aiohttp.ClientTimeout(total=10)
NO_TOKEN_WARNING = "No LOCAL_OCR_TOKEN set: anyone on your network can use this OCR server."


def load_or_create_token(path: str) -> str:
    """Return the persisted token, creating it (mode 0600) on first use."""
    p = Path(path)
    if p.is_file():
        existing = p.read_text().strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.chmod(p, 0o600)
    return token


async def _own_hostname(session: aiohttp.ClientSession, headers: dict[str, str]) -> str:
    # Supervisor renamed add-ons to apps in 2026; try the old path, fall back on 404.
    for path in ("addons", "apps"):
        async with session.get(f"{SUPERVISOR}/{path}/self/info", headers=headers, timeout=_TIMEOUT) as resp:
            if resp.status == 404 and path == "addons":
                continue
            resp.raise_for_status()
            return (await resp.json())["data"]["hostname"]
    raise aiohttp.ClientError("supervisor self info not found")


async def announce_supervisor(session: aiohttp.ClientSession, token: str, port: int) -> bool:
    """Tell Supervisor about this service so Home Assistant can offer the integration."""
    sup = os.environ.get("SUPERVISOR_TOKEN")
    if not sup:
        return False
    headers = {"Authorization": f"Bearer {sup}"}
    try:
        hostname = await _own_hostname(session, headers)
        body = {"service": "local_ocr", "config": {"host": hostname, "port": port, "token": token}}
        async with session.post(f"{SUPERVISOR}/discovery", json=body, headers=headers, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
    except (aiohttp.ClientError, TimeoutError, KeyError, TypeError, ValueError) as err:
        _LOGGER.warning("Supervisor discovery failed: %s", type(err).__name__)
        return False
    _LOGGER.info("Announced to Supervisor discovery as %s:%s", hostname, port)
    return True


def local_ipv4_addresses() -> list[str]:
    """Non-loopback IPv4 addresses of this host."""
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(str(info[4][0]))
    except OSError:
        pass
    try:  # the address used for the default route, even if the hostname resolves to loopback
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    return sorted({a for a in found if not a.startswith("127.") and a != "0.0.0.0"})  # noqa: S104


async def advertise_zeroconf(port: int, token_required: bool) -> AsyncZeroconf | None:
    """Standalone only: advertise `_local-ocr._tcp` on the LAN. Never raises."""
    try:
        addresses = local_ipv4_addresses()
        if not addresses:
            _LOGGER.warning("Zeroconf not started: no non-loopback IPv4 address")
            return None
        hostname = socket.gethostname()
        info = ServiceInfo(
            SERVICE_TYPE,
            f"Local OCR on {hostname}.{SERVICE_TYPE}",
            port=port,
            properties={"api": "1", "auth": "1" if token_required else "0"},
            addresses=[socket.inet_aton(a) for a in addresses],
            server=f"{hostname}.local.",
        )
        az = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await az.async_register_service(info)
    except Exception as err:  # noqa: BLE001 - discovery must never crash the server
        _LOGGER.warning("Zeroconf advertisement failed: %s", type(err).__name__)
        return None
    _LOGGER.info("Advertising %s on %s", SERVICE_TYPE, ", ".join(addresses))
    return az
