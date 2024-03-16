#!/bin/bash


#SBATCH --account=eccc_pegasus_mrd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=10G
#SBATCH --time=1:00:00


# Siqi Wei
# July 2023
# Script to run the case 6 to text EXODE



method=rk23
echo $method 
echo ${method^^}



