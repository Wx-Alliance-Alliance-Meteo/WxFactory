#!/bin/bash

#SBATCH --account=eccc_pegasus_mrd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --output=vicky/ADR_2D/run_scripts/ADR2D_work_precision_exode_%j.out
#SBATCH --error=vicky/ADR_2D/run_scripts/ADR2D_work_precision_exode_%j.err

python ADR_2D.py > ADR_2D_testoutput.txt
