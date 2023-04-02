#! /bin/bash -l

#SBATCH -J gef_p2304rk
#SBATCH -o gef_p2304rk.out
#SBATCH -e gef_p2304rk.error
#SBATCH -N 1  #how many cluster nodes #STEPHANE HAS TO CHANGE
#SBATCH -n 2304  #how many cores
#SBATCH --exclusive
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH -t 00:30:00
#SBATCH --array=1-7

conda activate gef3

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

cases=("case2.ini" "case2_kiopsne.ini" "case2_cwy1s.ini" "case2_cwyne.ini" "case2_cwyne1s.ini" "case2_icwy1s.ini" "case2_icwyne.ini" "case2_icwyne1s.ini" "case2_icwyiop.ini" "case2_pmex1s.ini" "case2_pmexne.ini" "case2_pmexne1.ini" "case6.ini" "case6_kiopsne.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_cwyne1s.ini" "case6_icwy1s.ine" "case6_icwyne.ini" "case6_icwyne1s.ini" "case6_icwyiop.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "case6_pmexne1s.ini" "case5.ini" "case5_kiopsne.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_cwyne1s.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_icwyne1s.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" "case5_pmexne1s.ini" "galewsky.ini" "galewsky_kiopsne.ini" "galewsky_cwy1s.ini" "galewsky_cwyne.ini" "galewsky_cwyne1s.ini" "galewsky_icwy1s.ini" "galewsky_icwyne.ini" "galewsky_icwyne1s.ini" "galewsky_icwyiop.ini" "galewsky_pmex1s.ini" "galewsky_pmexne.ini" "galewsky_pmexne1s.ini" "gaussian_bubble.ini" "gaussian_bubble_kiopsne.ini" "gaussian_bubble_cwy1s.ini" "gaussian_bubble_cwyne.ini" "gaussian_bubble_cwyne1s.ini" "gaussian_bubble_icwy1s.ini" "gaussian_bubble_icwyne.ini" "gaussian_bubble_icwyne1s.ini" "gaussian_bubble_icwyiop.ini" "gaussian_bubble_pmex1s.ini" "gaussian_bubble_pmexne.ini" "gaussian_bubble_pmexne1s.ini")

caselen=${#cases[@]}

for ((j = 0; j < $caselen; j++)); do
    mpirun -np 2304 python3 ./main_gef.py config/procs2304_epirk/${cases[$j]}
done
