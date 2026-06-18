#!/usr/bin/env bash

set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${workspace}/.venv_tf_wsl_gpu"
python_bin="${venv_path}/bin/python"
pip_bin="${venv_path}/bin/pip"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 nao encontrado no WSL."
  exit 1
fi

if [ ! -x "${python_bin}" ]; then
  python3 -m venv "${venv_path}"
fi

"${python_bin}" -m pip install --upgrade pip setuptools wheel
"${pip_bin}" install -r "${workspace}/requirements_tf_wsl_gpu.txt"

export PYTHONUNBUFFERED=1
export TF_XLA_FLAGS="--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
export TMPDIR="/tmp/treinamento_gpu_tmp"
export XDG_CACHE_HOME="${HOME}/.cache/treinamento_gpu"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/nv"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${CUDA_CACHE_PATH}"

# TensorFlow pip wheels ship CUDA/cuDNN libs inside site-packages; expose them explicitly.
nvidia_lib_dirs=("/usr/lib/wsl/lib")
shopt -s nullglob
for candidate in \
  "${venv_path}"/lib/python3.*/site-packages/nvidia/*/lib \
  "${venv_path}"/lib/python3.*/site-packages/nvidia/cuda_nvcc/nvvm/lib64; do
  if [ -d "${candidate}" ]; then
    nvidia_lib_dirs+=("${candidate}")
  fi
done
shopt -u nullglob
export LD_LIBRARY_PATH="$(IFS=:; echo "${nvidia_lib_dirs[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${python_bin}" "${workspace}/run_rerun_queue.py" \
  --workspace-root "${workspace}" \
  --python-bin "${python_bin}"
