#!/bin/bash

#SBATCH --account=eccc_mrd
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --job-name=adr2d_work_precision_5runs
#SBATCH --output=/home/siw001/WxFactory_EXODE/EXODE_projects/adr_2d/work_prec/%x_%j.out 
#SBATCH --error=/home/siw001/WxFactory_EXODE/EXODE_projects/adr_2d/work_prec/%x_%j.err

#conda activate gef 

epi_ord=5
exode_method="pmex"
#    "pmex",
#    "kiops",
#    "BS32",
#    "DP54",
#    "M43",
#    "KC32",
#    "EXLRK32",
#    "EXLRK43",
#    "ExLRK4(3)minA5",
#    "ExLRK4(3)minA5param0",
#    "ExLRK4(3)minA5param0d5",
#    "ExLRK4(3)minA5paramN0D25",
#    "Ralston43",

tol=1e-6
rol=1e-3

python adr_2d.py $epi_ord $exode_method $tol $rtol

