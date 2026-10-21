# Source this file in bash; set KUBOMSD_CUDA_ROOT for a different installation.
# Only the current shell is changed. No driver or system configuration changes.
export CUDA_PATH="${KUBOMSD_CUDA_ROOT:-/usr/local/cuda}"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
