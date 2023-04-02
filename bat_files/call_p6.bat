#! /bin/bash -l

#SBATCH -J gef_p96
#SBATCH -o file_t96.out
#SBATCH -e file_t96.error
#SBATCH -N 3  #how many cluster nodes
#SBATCH -n 96  #how many cores
#SBATCH -M merced
#SBATCH -p medium
#SBATCH -t 23:45:00
##SBATCH --array=1-2

conda activate gef3

#mpirun -np 6 python3 ./main_gef.py config/procs6_epirk/case6_pmex1s.ini

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

#cases=("case2.ini" "case2_cwy1s.ini" "case2_cwyne.ini" "case2_icwy1s.ini" "case2_icwyne.ini" "case2_icwyiop.ini" "case2_pmex1s.ini" "case2_pmexne.ini" "case6.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_icwy1s.ine" "case6_icwyne.ini" "case6_icwyiop.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "case5.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" )

#cases=("case6_kiopsne.ini" "case6_cwy1s.ini" "case6_pmex1s.ini" "galewsky_pmex1s.ini" "galewsky.ini")
cases=("case5_cwy1s.ini" "case5_pmex1s.ini" "case6_cwy1s.ini" "case6_pmex1s.ini" "galewsky_cwy1s.ini" "galewsky_pmex1s.ini")
caselen=${#cases[@]}

for ((j = 0; j < $caselen; j++)); do
    mpirun -np 96 python3 ./main_gef.py config/procs96/${cases[$j]}
done
