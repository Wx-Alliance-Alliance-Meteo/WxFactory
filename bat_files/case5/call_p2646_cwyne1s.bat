#! /bin/bash -l

#SBATCH -J gef_p2646
#SBATCH -o gef_p2646.out
#SBATCH -e gef_p2646.error
#SBATCH -N 34  #how many cluster nodes #STEPHANE HAS TO CHANGE
#SBATCH -n 2646  #how many cores
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH --array=1-7

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

mpirun -np 2646 python3 ./main_gef.py config/procs2646/case5_cwyne1s.ini
