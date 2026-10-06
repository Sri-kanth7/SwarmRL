#!/usr/bin/env bash
# Run only the smoke tests from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/smoke "$@"
