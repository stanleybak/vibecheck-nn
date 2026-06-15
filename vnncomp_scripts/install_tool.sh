#!/bin/bash
# VNNCOMP install_tool.sh for vibecheck.
#   args: <version>
# Installs a self-contained venv at $TOOL_DIR/.venv via uv (pulls a known-good
# Python + the torch wheels + gurobipy). Assumes a CUDA-capable host with an
# NVIDIA driver already present (the competition GPU image).
set -e

VERSION_STRING=v1
if [ "$1" != "${VERSION_STRING}" ]; then
	echo "Expected first argument (version string) '${VERSION_STRING}', got '$1'"
	exit 1
fi

TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")

# END-on-failure banner: `set -e` would otherwise abort mid-install with no
# closing marker. parse_log.py keys off these [vibecheck:install_tool] anchors.
T_START=$(date +%s.%N)
trap 'rc=$?; echo "================================================================"; echo "[vibecheck:install_tool] END status=fail rc=$rc"; echo "================================================================"' ERR

echo "================================================================"
echo "[vibecheck:install_tool] BEGIN version=$VERSION_STRING tool_dir=$TOOL_DIR"
echo "================================================================"

# psmisc provides killall (used by prepare_instance.sh).
if command -v apt-get >/dev/null 2>&1; then
	sudo -n apt-get update -y || true
	sudo -n apt-get install -y psmisc curl || true
fi

# Install uv if absent.
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

cd "$TOOL_DIR"
uv python install 3.12
uv venv --python 3.12 .venv
# gurobipy is in pyproject dependencies; the pip wheel's size-limited license
# is sufficient for the regular-track benchmarks (a full license is only needed
# for very large models). Install it explicitly here too in case of an older
# pyproject checkout.
VIRTUAL_ENV="$TOOL_DIR/.venv" uv pip install -e ".[dev]"
VIRTUAL_ENV="$TOOL_DIR/.venv" uv pip install gurobipy

trap - ERR
ELAPSED=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T_START}")
echo "================================================================"
echo "[vibecheck:install_tool] END status=ok elapsed=${ELAPSED}s"
echo "================================================================"
