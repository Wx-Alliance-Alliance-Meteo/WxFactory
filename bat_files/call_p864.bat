#! /bin/bash -l

#SBATCH -J gef_p864
#SBATCH -o gef_p864.out
#SBATCH -e gef_p864.error
#SBATCH -N 6  #how many cluster nodes #STEPHANE HAS TO CHANGE
#SBATCH -n 864  #how many cores
#SBATCH --exclusive
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH -t 03:00:00
#SBATCH --array=1-7

conda activate gef3

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

cases=("case6.ini" "case6_kiopsne.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_cwyne1s.ini" "case6_icwy1s.ine" "case6_icwyne.ini" "case6_icwyne1s.ini" "case6_icwyiop.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "case6_pmexne1s.ini" "case5.ini" "case5_kiopsne.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_cwyne1s.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_icwyne1s.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case5_pmexne1s.ini" "galewsky.ini" "galewsky_kiopsne.ini" "galewsky_cwy1s.ini" "galewsky_cwyne.ini" "galewsky_cwyne1s.ini" "galewsky_icwy1s.ini" "galewsky_icwyne.ini" "galewsky_icwyne1s.ini" "galewsky_icwyiop.ini" "galewsky_pmex1s.ini" "galewsky_pmexne.ini" "galewsky_pmexne1s.ini")

caselen=${#cases[@]}

for ((j = 0; j < $caselen; j++)); do
    mpirun -np 864 python3 ./main_gef.py config/procs864_epirk/${cases[$j]}
done
