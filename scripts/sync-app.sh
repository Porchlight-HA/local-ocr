#!/usr/bin/env bash
# Copy the server package into local_ocr/server/ so the App folder is self-contained.
# Supervisor builds a local App with the App folder as the Docker context, so it cannot see ../server.
# Allowlist only: pyproject.toml and local_ocr_server/ (no tests, build/, caches or .venv).
# local_ocr/server/ is gitignored; re-run this after changing the server and before building.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/local_ocr/server"
rm -rf "$dest"
mkdir -p "$dest"
cp "$root/server/pyproject.toml" "$dest/"
cp -R "$root/server/local_ocr_server" "$dest/local_ocr_server"
find "$dest" \( -name __pycache__ -o -name '*.pyc' \) -prune -exec rm -rf {} +
echo "synced server/ -> local_ocr/server/"
