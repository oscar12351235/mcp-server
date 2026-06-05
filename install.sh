#!/usr/bin/env bash
set -euo pipefail

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry is required but not installed." >&2
  exit 1
fi

export POETRY_VIRTUALENVS_IN_PROJECT=true

if command -v python3.12 >/dev/null 2>&1; then
  poetry env use python3.12 >/dev/null
else
  poetry env use python3 >/dev/null
fi

poetry install --with dev,lint

echo "Installed. Try: poetry run pytest"
