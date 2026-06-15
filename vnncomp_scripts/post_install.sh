#!/bin/bash
# VNNCOMP post-install script for vibecheck (OPTIONAL).
#
# Runs after install_tool.sh, on the real instance. Use it for machine-dependent
# setup that install_tool.sh can't do: here, installing the NVIDIA driver on a
# base Ubuntu AMI (e.g. Ubuntu Server 24.04) that ships WITHOUT one, and (later)
# any full-Gurobi-license activation.
#
# IMPORTANT: installing the driver requires a REBOOT for the kernel module to
# load. Enable the "restart after post-install" toggle in the submission form so
# the instance reboots before benchmark execution; this script is idempotent, so
# a working driver after reboot just short-circuits.
#
# Driver steps mirror the observed-good flow in AWS_SETUP.txt (newest headless
# `-server` driver, noninteractive).
set -u

echo "================================================================"
echo "[vibecheck:post_install] BEGIN"
echo "================================================================"

# Already have a working GPU driver? Nothing to do (also the post-reboot state).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
	echo "================================================================"
	echo "[vibecheck:post_install] END status=ok (driver already present)"
	echo "================================================================"
	exit 0
fi

echo "nvidia-smi not working: installing the NVIDIA driver..."
sudo apt-get update -y || true

# Pick the newest headless `-server` driver apt offers (AWS_SETUP.txt saw 580 on
# an A10G); fall back to the newest plain driver, then to ubuntu-drivers.
DRV=$(apt-cache search '^nvidia-driver-[0-9]' 2>/dev/null \
	| grep -oE 'nvidia-driver-[0-9]+-server' | sort -V | tail -1)
[ -z "$DRV" ] && DRV=$(apt-cache search '^nvidia-driver-[0-9]' 2>/dev/null \
	| grep -oE 'nvidia-driver-[0-9]+' | sort -V | tail -1)

if [ -n "$DRV" ]; then
	echo "Installing $DRV ..."
	sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$DRV"
elif command -v ubuntu-drivers >/dev/null 2>&1; then
	echo "No nvidia-driver-* package matched; trying 'ubuntu-drivers autoinstall'..."
	sudo DEBIAN_FRONTEND=noninteractive ubuntu-drivers autoinstall
else
	echo "ALARM: no nvidia-driver package and no ubuntu-drivers available."
	echo "[vibecheck:post_install] END status=fail (no driver package found)"
	exit 1
fi

echo "----------------------------------------------------------------"
echo "Driver installed. A REBOOT is required for the kernel module to load."
echo "Enable 'restart after post-install' in the submission form; after the"
echo "reboot, nvidia-smi (and prepare_instance.sh) will see the GPU."
echo "================================================================"
echo "[vibecheck:post_install] END status=ok (driver $DRV installed; reboot required)"
echo "================================================================"
exit 0
