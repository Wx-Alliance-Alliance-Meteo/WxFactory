#!/bin/bash -l

## Make sure SLURM parameters are correct (account, partition, etc.)

#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=hello
#SBATCH --output=hello.out
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=6
#SBATCH --mem-per-cpu=8000M

export SLURM_EXPORT_ENV=ALL

echo "hostname: $(hostname)"
#nvidia-smi
 
module load nvhpc-pmix/24.3
. /home/pr-h6bcfd5d/wx_env/bin/activate
module load nsight-systems/2023.3

cd ~/wx_factory

config_file=config/dcmip31_cuda_small.ini
[ $# -ge 1 ] && config_file=${1}

echo "Running WxFactory with config ${config_file}"
pwd

export MELLANOX_VISIBLE_DEVICES=0
#export UCX_UD_TIMEOUT=120s
#export UCX_TLS=^ud
export UCX_TLS=^ud,ud:aux
#mpirun -n 6 --map-by ppr:4:socket:PE=16 ./main_gef.py ${config_file}
mpirun -n 6 --map-by ppr:4:socket:PE=16 nsys profile -o gef_tmp.%q{OMPI_COMM_WORLD_RANK} -f true -t cuda,nvtx ./main_gef.py ${config_file}
#mpirun -np 6  --tag-output  --map-by ppr:4:socket:PE=16    numactl --show 
#mpirun -np 1  --tag-output  --map-by ppr:4:socket:PE=16    numactl --show 
#nvidia-smi topo -m

#lscpu

