
# MPI
. /fs/ssm/main/opt/intelcomp/master/inteloneapi_2022.1.2_multi/oneapi/compiler/latest/env/vars.sh
. /fs/ssm/main/opt/intelcomp/master/inteloneapi_2022.1.2_multi/oneapi/mpi/latest/env/vars.sh

# Python
source /home/vma000/launch-scripts/conda/load_u2_env.sh
conda activate gef

# Environment variables
# export LANG=en_CA.UTF-8
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen

