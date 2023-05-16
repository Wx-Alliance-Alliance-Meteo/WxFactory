#!/bin/bash

#call the slurm scripts for different processors

#run with epi6
sbatch bat_files/call_p6.bat

#process_idp6=$!
#wait $process_idp6

#run with srerk
sbatch bat_files/call_p6rk.bat

