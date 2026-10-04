#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
"$PY" analysis/release_gate.py
"$PY" scripts/grade.py
"$PY" scripts/make_eval_set.py --self-check
"$PY" scripts/test_release_verdict.py
"$PY" scripts/test_qualify_e2e.py
"$PY" scripts/test_qualify_matrix_e2e.py
"$PY" scripts/test_ci_action.py
echo "All synthetic gate checks passed"
