#!/usr/bin/env bash
# Build the MAPPO training configuration (Week 1: no training runs).
# Windows/PowerShell equivalent:
#   python -m training.train_mappo --print-config
set -euo pipefail
cd "$(dirname "$0")/.."
python -m training.train_mappo "$@"
