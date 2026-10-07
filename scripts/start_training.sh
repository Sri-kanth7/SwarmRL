#!/usr/bin/env bash
# Build the MAPPO training configuration (MAPPO stays configuration-only).
# Windows/PowerShell equivalent:
#   python -m training.train_mappo --print-config
# IPPO training runs through:
#   python -m training.train_ippo --config training/configs/ippo.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
python -m training.train_mappo "$@"
