#!/usr/bin/env bash
set -e

# Canonical baseline training entrypoint.
# Requires processed artifacts to exist. Run preprocess first if needed:
#   bash scripts/run_preprocess.sh
python -m src.training.baseline_runner "$@"
