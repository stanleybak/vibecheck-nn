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
echo "==> system packages (psmisc, curl) via apt, if available"
if command -v apt-get >/dev/null 2>&1; then
	sudo -n apt-get update -y || true
	sudo -n apt-get install -y psmisc curl || true
else
	echo "    apt-get not present; skipping"
fi

echo "==> ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
else
	echo "    uv already present ($(uv --version))"
fi

cd "$TOOL_DIR"
echo "==> python 3.12 + venv at $TOOL_DIR/.venv"
uv python install 3.12
# --clear so a re-run (or a partial earlier install) gets a clean venv rather
# than erroring on an existing .venv.
uv venv --clear --python 3.12 .venv
# gurobipy is in pyproject dependencies; the pip wheel's size-limited license
# is sufficient for the regular-track benchmarks (a full license is only needed
# for very large models). Install it explicitly here too in case of an older
# pyproject checkout.
echo "==> installing vibecheck + dependencies (torch, gurobipy, onnxsim, ...)"
VIRTUAL_ENV="$TOOL_DIR/.venv" uv pip install -e ".[dev]"
VIRTUAL_ENV="$TOOL_DIR/.venv" uv pip install gurobipy

# Proof-of-success in the install log (a failed import here trips the ERR trap
# above, so a broken env surfaces as END status=fail rather than silently).
echo "==> verifying install"
"$TOOL_DIR/.venv/bin/python" - <<'PY'
import platform, torch, gurobipy, vibecheck
print(f"    python   {platform.python_version()}")
print(f"    torch    {torch.__version__}  (cuda build: {torch.version.cuda})")
print(f"    gurobipy {'.'.join(map(str, gurobipy.gurobi.version()))}")
print("    vibecheck import OK")
PY

trap - ERR
ELAPSED=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T_START}")
echo "================================================================"
echo "[vibecheck:install_tool] END status=ok elapsed=${ELAPSED}s"
echo "================================================================"
