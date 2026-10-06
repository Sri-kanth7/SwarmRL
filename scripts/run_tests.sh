#!/usr/bin/env bash
# Run the full pytest suite from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest "$@"
