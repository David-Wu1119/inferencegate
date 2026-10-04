#!/usr/bin/env bash
# Deterministic planted scenarios; temporary mock records, no model or GPU.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONDONTWRITEBYTECODE=1 "${PYTHON:-python3}" scripts/test_qualify_e2e.py
