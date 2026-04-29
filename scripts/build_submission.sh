#!/usr/bin/env bash
# Build the submission zip: RLI_22_00 - Group XX.zip
#
# Usage: GROUP=XX make zip   (or:  bash scripts/build_submission.sh XX)
#
# Includes: src/, the notebook, requirements.txt, README, Makefile, .python-version,
# pyproject.toml, the cached artifacts/results/ + artifacts/figures/.
# Excludes: .venv, __pycache__, tb_logs, checkpoints (large + regenerable), .git.

set -euo pipefail

GROUP="${1:-XX}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="RLI_22_00 - Group ${GROUP}"
OUT="${ROOT}/${NAME}.zip"
STAGE="$(mktemp -d)/${NAME}"

echo "==> Staging submission at: ${STAGE}"
mkdir -p "${STAGE}"

# Copy package + notebook + docs + scripts + meta
cp -R "${ROOT}/src"          "${STAGE}/"
cp -R "${ROOT}/scripts"      "${STAGE}/"
cp -R "${ROOT}/docs"         "${STAGE}/"
cp    "${ROOT}/Makefile"     "${STAGE}/"
cp    "${ROOT}/pyproject.toml" "${STAGE}/"
cp    "${ROOT}/requirements.txt" "${STAGE}/"
cp    "${ROOT}/README.md"    "${STAGE}/"
cp    "${ROOT}/.python-version" "${STAGE}/"

# Cached artifacts (results JSON + figures PNG only — not tb_logs / checkpoints)
mkdir -p "${STAGE}/artifacts"
[ -d "${ROOT}/artifacts/results" ] && cp -R "${ROOT}/artifacts/results" "${STAGE}/artifacts/"
[ -d "${ROOT}/artifacts/figures" ] && cp -R "${ROOT}/artifacts/figures" "${STAGE}/artifacts/"

# Clean up __pycache__ / .ipynb_checkpoints from staged copy
find "${STAGE}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGE}" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

echo "==> Zipping to: ${OUT}"
rm -f "${OUT}"
( cd "$(dirname "${STAGE}")" && zip -r "${OUT}" "${NAME}" >/dev/null )

echo "==> Done."
ls -lh "${OUT}"
