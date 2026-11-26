#!/usr/bin/env bash
# Run like run.sh, with the ranks sharing each GPU through NVIDIA MPS (Multi-Process Service).
#
#   ./scripts/run_mps.sh -n 6 ./WxFactory config/dcmip21.ini
#
# Without MPS, ranks that share a GPU take turns on it (time slicing), so each rank's halo exchange mostly waits for
# the other ranks' GPU work. MPS lets their kernels run concurrently; with several ranks per GPU this was 1.25 to 1.5
# times faster. With one rank per GPU it changes nothing.
#
# The script uses a private MPS daemon (pipe and log directories below), started here and stopped on exit; a daemon
# already listening on that directory is reused and left running. Caveat: under MPS, a GPU out-of-memory error can
# show up as a hang instead of an error, so a run that stalls at once is likely out of GPU memory.

set -euo pipefail

mps_root="${XDG_RUNTIME_DIR:-$HOME/.cache}/wxfactory-mps"
export CUDA_MPS_PIPE_DIRECTORY="$mps_root/pipe"  # a short path: it holds a Unix socket
export CUDA_MPS_LOG_DIRECTORY="$mps_root/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

if [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] && echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "Using the MPS daemon already running in $mps_root"
else
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
fi

# run.sh does not export CUDA_* variables, so pass the MPS directories explicitly.
"$(dirname "$0")/run_mpi.sh" -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY "$@"
