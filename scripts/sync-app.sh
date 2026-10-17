#!/usr/bin/env bash
# Copy server/ into local_ocr/server/ so the App folder is self-contained.
# Supervisor builds a local App with the App folder as the Docker context, so it cannot see ../server.
# local_ocr/server/ is gitignored; re-run this after changing the server and before building.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/local_ocr/server"
rm -rf "$dest"
mkdir -p "$dest"
tar -C "$root/server" \
  --exclude='tests' --exclude='scripts' --exclude='.venv' --exclude='__pycache__' \
  --exclude='*.egg-info' --exclude='.pytest_cache' --exclude='*.pyc' \
  -cf - . | tar -C "$dest" -xf -
echo "synced server/ -> local_ocr/server/"
