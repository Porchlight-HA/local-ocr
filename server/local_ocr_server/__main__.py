"""Run the Local OCR server: python -m local_ocr_server --port 8738"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

import aiohttp
from aiohttp import web

from . import __version__
from .app import create_app
from .discovery import NO_TOKEN_WARNING, advertise_zeroconf, announce_supervisor, load_or_create_token
from .engine import OcrEngine
from .limits import effective_cpus


def load_token(token_file: str | None) -> str | None:
    """LOCAL_OCR_TOKEN wins, else the token file's contents if it exists, else no auth."""
    env = os.environ.get("LOCAL_OCR_TOKEN")
    if env:
        return env
    if token_file:
        path = Path(token_file)
        if path.is_file():
            return path.read_text().strip() or None
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="local_ocr_server")
    p.add_argument("--port", type=int, default=8738)
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container service
    p.add_argument("--token-file")
    p.add_argument("--idle-unload-minutes", type=float, default=10)
    p.add_argument("--max-side", type=int, default=2000)
    p.add_argument("--standalone", action="store_true", help="run outside Home Assistant: zeroconf instead of Supervisor discovery")
    return p


def make_after_listen(app: web.Application, managed: bool, token: str | None, port: int):
    """Return the coroutine function that announces the service once the server is listening."""
    log = logging.getLogger(__name__)
    if managed:

        async def announce(_app: web.Application) -> None:
            async with aiohttp.ClientSession() as session:
                await announce_supervisor(session, token or "", port)

        return announce

    async def advertise(app: web.Application) -> None:
        app["zeroconf"] = await advertise_zeroconf(port, token_required=token is not None)

    async def unadvertise(app: web.Application) -> None:
        az = app.get("zeroconf")
        if az is not None:
            try:
                await az.async_unregister_all_services()
                await az.async_close()
            except Exception:  # noqa: BLE001
                log.warning("Zeroconf shutdown failed")

    app.on_cleanup.append(unadvertise)
    return advertise


async def serve(app, host, port, after_listen, stop: asyncio.Event | None = None) -> None:
    """Run the app; start discovery only after the site is accepting connections."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    task = None
    try:
        await web.TCPSite(runner, host, port).start()
        task = asyncio.create_task(after_listen(app))
        stop = stop or asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        await stop.wait()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = OcrEngine(
        threads=effective_cpus(),
        idle_unload_s=args.idle_unload_minutes * 60,
        max_side=args.max_side,
    )
    managed = bool(os.environ.get("SUPERVISOR_TOKEN")) and not args.standalone
    if managed:
        token: str | None = load_or_create_token(args.token_file or "/data/token")
    else:
        token = load_token(args.token_file)
        if token is None and args.host == "0.0.0.0":  # noqa: S104
            logging.getLogger(__name__).warning(NO_TOKEN_WARNING)
    app = create_app(engine, token, __version__)
    app["standalone"] = not managed

    after_listen = make_after_listen(app, managed, token, args.port)
    asyncio.run(serve(app, args.host, args.port, after_listen))


if __name__ == "__main__":
    main()
