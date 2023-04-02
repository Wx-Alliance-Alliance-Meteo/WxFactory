#! /bin/bash -l

#SBATCH -J gef_p294
#SBATCH -o gef_p294.out
#SBATCH -e gef_p294.error
#SBATCH -N 4  #how many cluster nodes 
#SBATCH -n 294  #how many cores
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH --array=1-7


#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

mpirun -np 294 python3 ./main_gef.py config/procs294/case5_pmex1s.ini
