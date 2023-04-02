#! /bin/bash -l

#SBATCH -J gef
#SBATCH -o file_kiops.out
#SBATCH -e file_kiops.error
#SBATCH -N 1  #how many cluster nodes
#SBATCH -n 24  #how many cores
#SBATCH -M merced
##SBATCH --exclusive
##SBATCH --exclude=mrcd[30-93]
##SBATCH --exclude=mrcd[26-27,29,30-60]
##SBATCH --exclude=mrcd[33-60]
#SBATCH -p short
##SBATCH -w mrcd[95-110,114]
##SBATCH --constraint=ib
#SBATCH -t 01:30:00
##SBATCH --array=1-2

#was using original gef conda env
conda activate gef3

mpirun -np 24 python3 ./main_gef.py config/gaussian_bubble.ini


#cases=("case2.ini" "case2_cwy1s.ini" "case2_cwyne.ini" "case2_icwy1s.ini" "case2_icwyne.ini" "case2_icwyiop.ini" "case2_pmex1s.ini" "case2_pmexne.ini" "case6.ini" "case6_cwy1s.ini" "case6_cwyne.ini" "case6_icwy1s.ine" "case6_icwyne.ini" "case6_icwyiop.ini" "case6_pmex1s.ini" "case6_pmexne.ini" "case5.ini" "case5_cwy1s.ini" "case5_cwyne.ini" "case5_icwy1s.ini" "case5_icwyne.ini" "case5_icwyiop.ini" "case5_pmex1s.ini" "case5_pmexne.ini" )

#cases=("case2.ini" "case2_icwyiop.ini")
#caselen=${#cases[@]}

#for ((j = 0; j < $caselen; j++)); do
#    mpirun -np 24 python3 ./main_gef.py config/${cases[$j]}
#done
