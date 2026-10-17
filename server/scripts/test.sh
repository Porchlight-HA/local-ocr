#!/usr/bin/env bash
# Run the server test suite on Linux (python:3.13-slim-trixie), matching the production OS and Python.
# Usage: scripts/test.sh [pytest args]   e.g. scripts/test.sh -q -s tests/test_golden.py
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # local-ocr/
exec docker run --rm \
  -v "$here":/work \
  -v local-ocr-pip-cache:/root/.cache/pip \
  -w /work/server \
  -e PYTHONDONTWRITEBYTECODE=1 \
  python:3.13-slim-trixie \
  sh -c 'pip install -q -e ".[test]" 2>&1 | grep -v "root. user" ; exec python -m pytest "$@"' sh "$@"
