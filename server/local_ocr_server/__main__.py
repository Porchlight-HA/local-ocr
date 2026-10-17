"""Run the Local OCR server: python -m local_ocr_server --port 8738"""
from __future__ import annotations

import argparse
import logging
import os
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

    if managed:

        async def _announce(_app: web.Application) -> None:
            async with aiohttp.ClientSession() as session:
                await announce_supervisor(session, token, args.port)

        app.on_startup.append(_announce)
    else:

        async def _advertise(app: web.Application) -> None:
            app["zeroconf"] = await advertise_zeroconf(args.port, token_required=token is not None)

        async def _unadvertise(app: web.Application) -> None:
            az = app.get("zeroconf")
            if az is not None:
                try:
                    await az.async_unregister_all_services()
                    await az.async_close()
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning("Zeroconf shutdown failed")

        app.on_startup.append(_advertise)
        app.on_cleanup.append(_unadvertise)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
