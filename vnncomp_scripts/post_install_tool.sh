#!/bin/bash

# Script for setting up Gurobi after running install_tool.sh.
# This gets manually pasted into the VNNCOMP web form during the competition
# (with the real license instead of the placeholder below). It is NOT committed
# with a real license -- this repo copy is a template.
#
# It requires first generating a node-locked academic license:
#   1. Run install_tool.sh with VIBECHECK_GUROBI_PROBE=1 on an instance with the
#      licensing ENI attached (MAC 02:72:ca:20:72:87 -> Gurobi HOSTID ca207287).
#      It prints a Gurobi keyserver URL.
#   2. Open that URL from a UNIVERSITY network to mint the license.
#   3. Paste the returned license text between the EOF markers below.
# Because the fixed ENI pins the MAC -> HOSTID, the same license is reused on
# every run; regenerate only if the ENI/MAC changes or the license expires.
#
# Heavily based on alpha-beta-CROWN's post_install_tool.sh:
#   https://github.com/Verified-Intelligence/alpha-beta-CROWN_vnncomp2024/blob/master/vnncomp_scripts/post_install_tool.sh
set -u

echo "================================================================"
echo "[vibecheck:post_install] BEGIN gurobi license"
echo "================================================================"

# Get the license from the URL which is the final output of install_tool.sh.
# The URL should be visited from a university network.
# Update the license below when the evaluation is paused.
cat > /tmp/gurobi.lic <<'EOF'
# TODO (submission): paste the academic node-locked gurobi.lic here. It should
# look like the following, with HOSTID=ca207287 (our licensing ENI) and VERSION=13:
#
# TYPE=ACADEMIC
# VERSION=13
# HOSTNAME=ip-...
# HOSTID=ca207287
# USERNAME=ubuntu
# EXPIRATION=YYYY-MM-DD
# KEY=...
# CKEY=...
EOF

# /opt/gurobi/gurobi.lic is a default search location found by any user that
# runs the benchmarks. Needs root: enable "run post-install as root" in the form
# if the image lacks passwordless sudo.
sudo -n mkdir -p /opt/gurobi && sudo -n mv /tmp/gurobi.lic /opt/gurobi/gurobi.lic \
	&& echo "license moved to /opt/gurobi/gurobi.lic" \
	|| echo "WARNING: could not write /opt/gurobi/gurobi.lic (sudo? run-post-install-as-root?)"

# Verify the FULL license is active: a >2000-var model fails on the bundled
# size-limited license, so this distinguishes "full license working" from
# "silently fell back to the limited license".
TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
PY="${VNNCOMP_PYTHON_PATH:-$TOOL_DIR/.venv/bin}/python"
status=ok
GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic "$PY" - <<'PY' || status=fail
import sys
import gurobipy as gp
try:
    e = gp.Env(); m = gp.Model(env=e); m.Params.OutputFlag = 0
    m.addVars(3000); m.optimize()
    print("  full Gurobi license ACTIVE (3000-var model solved)")
except gp.GurobiError as ex:
    print(f"  WARNING: full license NOT active: {ex}")
    sys.exit(1)
PY

echo "================================================================"
echo "[vibecheck:post_install] END status=$status"
echo "================================================================"
exit 0
