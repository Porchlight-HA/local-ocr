#!/usr/bin/env bash
# Smoke test for a built Local OCR image.
# Usage: tests/smoke.sh <image> [size-budget-bytes]
# Starts the container in standalone mode, waits for /v1/health, reads a synthetic bulk-mail
# envelope, asserts generic_addressee is true, and checks the image size budget.
set -euo pipefail

image="${1:?usage: tests/smoke.sh <image> [size-budget-bytes]}"
budget="${2:-262144000}" # 250 MB, compressed (gzip of docker save)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$here/fixtures/envelopes/env02_id.jpg"
name="local-ocr-smoke-$$"
port="${SMOKE_PORT:-18738}"
rc=0

cleanup() {
  if [ "$rc" -ne 0 ]; then docker logs "$name" 2>&1 | tail -n 30 || true; fi
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap 'rc=$?; cleanup; exit $rc' EXIT

command -v jq >/dev/null || { echo "FAIL: jq is required" >&2; exit 1; }
docker image inspect "$image" >/dev/null 2>&1 || { echo "FAIL: image $image not found" >&2; exit 1; }

# Plain docker run: /init starts the s6 service, which falls back to standalone mode without SUPERVISOR_TOKEN.
docker run -d --name "$name" -p "127.0.0.1:$port:8738" "$image" >/dev/null

running() { [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ]; }

ready=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/health" >/dev/null 2>&1; then ready=1; break; fi
  running || { echo "FAIL: the container exited before /v1/health came up" >&2; exit 1; }
  sleep 1
done
[ "$ready" = 1 ] || { echo "FAIL: /v1/health did not come up within 60 s" >&2; exit 1; }

# The first read loads the model; 120 s is far above a cold load on a CI runner.
resp="$(curl -fsS --max-time 120 -F "image=@$fixture" -F preset=envelope "http://127.0.0.1:$port/v1/read")" \
  || { echo "FAIL: /v1/read failed or took longer than 120 s" >&2; exit 1; }
got="$(printf '%s' "$resp" | jq -r '.fields.generic_addressee')"
[ "$got" = "true" ] || { echo "FAIL: fields.generic_addressee = $got, expected true" >&2; exit 1; }

size="$(docker save "$image" | gzip -c | wc -c | tr -d ' ')"
echo "compressed image size: $size bytes (budget $budget)"
[ "$size" -le "$budget" ] || { echo "FAIL: image size $size exceeds budget $budget" >&2; exit 1; }

echo OK
