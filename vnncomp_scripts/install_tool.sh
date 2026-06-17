#!/bin/bash
# VNNCOMP install_tool.sh for vibecheck.
#   args: <version>
# On a base Ubuntu Server 24.04 GPU AMI this: cleans up apt (purges the
# unattended-upgrades/snapd lock-holders), installs the pinned NVIDIA driver via
# the .run installer and loads it WITHOUT a reboot, then builds a self-contained
# venv at $TOOL_DIR/.venv via uv (Python 3.12 + torch + gurobipy). No CUDA
# toolkit/nvcc is installed - torch bundles its own CUDA runtime.
set -e

VERSION_STRING=v1
if [ "$1" != "${VERSION_STRING}" ]; then
	echo "Expected first argument (version string) '${VERSION_STRING}', got '$1'"
	exit 1
fi

# Sanity check: this install assumes the VNNCOMP AWS image's default 'ubuntu'
# user (its home, passwordless sudo, file ownership), and does system-level work
# (apt purge, NVIDIA driver). 'root' is also fine (the optional run-install-as-
# root mode). Refuse to run as anyone else so it can't touch a dev box by
# accident; bypass with VIBECHECK_SKIP_USER_CHECK=1.
_who=$(id -un)
if [ "$_who" != "ubuntu" ] && [ "$_who" != "root" ] && [ "${VIBECHECK_SKIP_USER_CHECK:-0}" != "1" ]; then
	echo "Refusing to install as user '$_who': this script does system-level setup"
	echo "(apt purge, NVIDIA driver install) and assumes the VNNCOMP AWS image's"
	echo "'ubuntu' user (its home dir for the uv install, plus passwordless sudo for"
	echo "the privileged steps). 'root' (run-install-as-root) is also accepted."
	echo "Set VIBECHECK_SKIP_USER_CHECK=1 to override (e.g. on a dev box)."
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

# --- apt hygiene + build deps -------------------------------------------------
# unattended-upgrades / snapd hold the apt lock on a fresh AWS image and cause
# spurious install failures, so stop+purge them first (mirrors alpha-beta-CROWN).
# We deliberately do NOT 'apt upgrade': a kernel bump would force a reboot before
# the dkms NVIDIA module could load, and we want the driver usable WITHOUT a
# reboot. dkms + build-essential + headers-for-the-RUNNING-kernel build the
# module; aria2 speeds the driver download. psmisc provides killall for
# prepare_instance.sh. No CUDA toolkit/nvcc - torch ships its own runtime.
echo "==> apt hygiene + build deps"
if command -v apt-get >/dev/null 2>&1; then
	export DEBIAN_FRONTEND=noninteractive
	sudo -n systemctl stop unattended-upgrades 2>/dev/null || true
	sudo -n killall -9 unattended-upgrade-shutdown 2>/dev/null || true
	sudo -n apt-get purge -y snapd unattended-upgrades modemmanager 2>/dev/null || true
	sudo -n apt-get update -y || true
	sudo -n apt-get install -y psmisc curl wget aria2 build-essential dkms "linux-headers-$(uname -r)" \
		|| sudo -n apt-get install -y psmisc curl wget aria2 build-essential dkms || true
else
	echo "    apt-get not present; skipping (driver install unavailable)"
fi

# --- NVIDIA driver ------------------------------------------------------------
# Pinned .run installer + dkms, loaded in place (rmmod + modprobe) so NO reboot
# is needed. 580.159.03 supports both the vibecheck venv (torch cu130) and abc
# (torch cu128) on the A10G. Gated: default 'auto' installs ONLY if nvidia-smi is
# not already working; VIBECHECK_INSTALL_DRIVER=0 skips, =1 forces. Override the
# version with VIBECHECK_DRIVER_VERSION. Failures only WARN (the toolkit install
# still succeeds; prepare_instance.sh's GPU alarm flags a missing driver later).
DRIVER_VERSION="${VIBECHECK_DRIVER_VERSION:-580.159.03}"
_install_nvidia_driver() {
	local run="/tmp/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
	local url="https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
	echo "    downloading driver ${DRIVER_VERSION}"
	if command -v aria2c >/dev/null 2>&1; then
		aria2c -x 10 -s 10 -k 1M -d /tmp -o "NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" "$url" || { echo "    download failed"; return 1; }
	else
		wget -q -O "$run" "$url" || { echo "    download failed"; return 1; }
	fi
	chmod +x "$run"
	sudo -n nvidia-smi -pm 0 2>/dev/null || true
	echo "    running .run installer (--silent --dkms)"
	sudo -n "$run" --silent --dkms || { echo "    driver installer failed"; rm -f "$run"; return 1; }
	# Load the freshly built module in place (no reboot): drop any stale modules,
	# then modprobe the new one and turn on persistence mode.
	sudo -n rmmod nvidia_uvm 2>/dev/null || true
	sudo -n rmmod nvidia_drm 2>/dev/null || true
	sudo -n rmmod nvidia_modeset 2>/dev/null || true
	sudo -n rmmod nvidia 2>/dev/null || true
	sudo -n modprobe nvidia 2>/dev/null || true
	sudo -n nvidia-smi -pm 1 2>/dev/null || true
	rm -f "$run"
	if nvidia-smi >/dev/null 2>&1; then
		echo "    driver active (loaded without reboot)"
		nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
		return 0
	fi
	echo "    driver installed but NOT active; a REBOOT may be required"
	echo "    (enable the restart toggle in the submission form as a fallback)."
	return 1
}

echo "==> NVIDIA driver (${DRIVER_VERSION})"
_drvmode="${VIBECHECK_INSTALL_DRIVER:-auto}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
	echo "    already working; skipping"
	nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
elif [ "$_drvmode" = "0" ]; then
	echo "    not working, but VIBECHECK_INSTALL_DRIVER=0; skipping"
elif command -v apt-get >/dev/null 2>&1; then
	_install_nvidia_driver || echo "    WARNING: driver setup incomplete; prepare_instance.sh's GPU alarm will flag this"
else
	echo "    WARNING: no apt-get; cannot install NVIDIA driver"
fi

# --- vibecheck venv -----------------------------------------------------------
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
# gurobipy is in pyproject dependencies; the pip wheel's size-limited license is
# sufficient for the regular-track benchmarks (a full license is only needed for
# very large models). Install it explicitly here too in case of an older
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

# --- Gurobi license probe (provisioning only; gated) --------------------------
# Only runs with VIBECHECK_GUROBI_PROBE=1, so normal eval installs (and
# non-Gurobi benchmarks like acasxu) don't download the 62 MB Gurobi tools or
# print a spurious hostid-mismatch warning. Enable it for the one-time license-
# provisioning submission: it fetches grbprobe (pip gurobipy ships none), prints
# the node-locked HOSTID/HOSTNAME/USERNAME/CORES and a keyserver URL you open
# from a university network to mint the .lic, which you paste into
# post_install_tool.sh and upload as the post-install script in the VNNCOMP form.
if [ "${VIBECHECK_GUROBI_PROBE:-0}" = "1" ]; then
	echo "==> Gurobi license probe"
	GRB_BIN=/tmp/gurobi1302/linux64/bin
	if [ ! -x "$GRB_BIN/grbprobe" ]; then
		echo "Fetching grbprobe..."
		curl -sL -o /tmp/gurobi1302.tar.gz https://packages.gurobi.com/13.0/gurobi13.0.2_linux64.tar.gz \
			&& tar xzf /tmp/gurobi1302.tar.gz -C /tmp \
				gurobi1302/linux64/bin/grbprobe gurobi1302/linux64/bin/gurobi_cl \
			|| echo "WARNING: grbprobe fetch failed; skipping license probe"
	fi

	grbprobe_output=$("$GRB_BIN/grbprobe" 2>/dev/null)
	echo "$grbprobe_output"

	HOSTNAME=$(echo $grbprobe_output | grep -Po "(?<=HOSTNAME=)(.*?)(?= )")
	HOSTID=$(echo $grbprobe_output | grep -Po "(?<=HOSTID=)(.*?)(?= )")
	USERNAME=$(echo $grbprobe_output | grep -Po "(?<=USERNAME=)(.*?)(?= )")
	CORES=$(echo $grbprobe_output | grep -Po "(?<=CORES=)(.*?)(?= )")
	LOCALDATE=$(date -u +%F)

	# The 2026 VNNCOMP vibecheck eval uses a fixed licensing ENI (MAC
	# 02:72:ca:20:72:87) -> hostid must be ca207287.
	EXPECTED_HOSTID=ca207287
	if [ "$HOSTID" != "$EXPECTED_HOSTID" ]; then
		echo "WARNING: HOSTID=$HOSTID does not match the expected ENI hostid ($EXPECTED_HOSTID) for the 2026 VNNCOMP vibecheck eval -- the license in post_install_tool.sh will NOT match this machine."
	fi

	echo "Obtain an academic KEY: https://portal.gurobi.com/iam/licenses/request/?type=academic"
	KEY=to-be-filled
	# Open this URL from a university network to mint the node-locked license:
	probe_url="https://portal.gurobi.com/keyserver?id=${KEY}&hostname=${HOSTNAME}&hostid=${HOSTID}&username=${USERNAME}&os=linux&localdate=${LOCALDATE}&version=13&cores=${CORES}"
	echo "$probe_url"
fi

ELAPSED=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T_START}")
echo "================================================================"
echo "[vibecheck:install_tool] END status=ok elapsed=${ELAPSED}s"
echo "================================================================"
