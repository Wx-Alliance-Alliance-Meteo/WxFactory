#! /bin/bash -l

#SBATCH -J newGef
#SBATCH -o file_kiops.out
#SBATCH -e file_kiops.error
#SBATCH -M merced #using merced cluster
#SBATCH -N 1  #how many cluster nodes
#SBATCH -n 24  #how many cores
#SBATCH --partition test
##SBATCH -w mrcd[95-110,114]
##SBATCH --constraint=ib
#SBATCH -t 00:30:00

module load anaconda3
conda activate gef3

#srun --mpi=pmi2 --cpu-bind=cores python3 laplacian_exp_perf.py 5
mpirun -np 24 python3 ./main_gef.py config/case2.ini


