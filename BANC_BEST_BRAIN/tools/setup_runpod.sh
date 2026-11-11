#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu GPU machine with CUDA (e.g. runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404):
# toolchain, then the Python packages pinned in BANC_BEST_BRAIN/requirements.txt (the versions of all results).
# LD_LIBRARY_PATH may be unset on a fresh shell, so it is referenced with a default.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential cmake git rsync > /dev/null
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
grep -q 'cuda/bin' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
nvcc --version | tail -n 1
apt-get install -y -qq python3-venv > /dev/null
[ -d /workspace/venv ] || python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install -q --upgrade pip
pip install -q -r "$HERE/../requirements.txt"
python - <<'EOF'
import brian2, brian2cuda, numpy
print("brian2", brian2.__version__, " brian2cuda", brian2cuda.__version__, " numpy", numpy.__version__)
EOF
grep -q 'workspace/venv' ~/.bashrc || echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
mkdir -p /workspace/sim
echo "setup done"
