#!/usr/bin/env bash

#PBS -N case6_strong_scaling_big_job_kiops
#PBS -l select=133:ncpus=80:mem=400gb:mpiprocs=80:ompthreads=1
#PBS -l walltime=3:00:00 
#PBS -e ./testoutput/work_precision/case6_5x672/
#PBS -o ./testoutput/work_precision/case6_5x672/



# Script to run the case 6 to text EXODE

start_time=$(date +%s)


# Load GEF runtime environment
#eval "$(/fs/homeu2/eccc/mrd/ords/rpnatm/siw001/miniconda3/bin/conda shell.bash hook)"
#conda activate gef 

# Load WX PIP runtime environment
. /space/hall5/sitestore/eccc/mrd/rpnatm/siw001/python-environments/wx/bin/activate
. r.load.dot rpn/code-tools/20240719/env/inteloneapi-2022.1.2

# Save the current directory location
startingDir=$PBS_O_WORKDIR
cd ${startingDir}

# Bubble location
GEFDir=$startingDir

# Config file location
configDir=$GEFDir/config

EPImethod_sel="epi5"
#epi_stiff3 epi_stiff4 epi_stiff5 epi_stiff6"

# config file name
configFile=case6_exode.ini

# testing stepsizes
stepsize="900"
output_freq=4

#spacial_dimension
nbsolpts=5
nb_elements_sel="672" #672 672 672
#nb_elements=672
num_of_cores="10584"
#"864 1176 2646 4704 10584"
tf=14400
#14400

tol_sel="1e-6"
#1e-12" # 1e-7 1e-8 1e-9 1e-10 1e-11 1e-12"

# EXODE methods 
exodemethod="BS32" 
#'BS32','DP5(4)', 'M4(3)','KC3(2)','EXLRK3(2)','EXLRK4(3)','F14(12)','DP8(7)','F10(8)'

for EPImethod in $EPImethod_sel 
do
	# Testoutput location and file
	testoutputDir=/space/hall6/sitestore/eccc/mrd/rpnatm/siw001/testoutput/case6/work_precision/
	mkdir -p $testoutputDir
for tol in $tol_sel 
do	
	for dt in $stepsize
	do
    
	    for core in $num_of_cores
	    do
		
		for exode_method in $exodemethod
		do
			for  nb_elements in $nb_elements_sel
                        do
			# Testoutput location and file
			testoutputSubDir=${testoutputDir}/case6_${nbsolpts}x${nb_elements}/${exode_method}/dt_${dt}
			mkdir -p ${testoutputSubDir}
			testoutputFile=${exode_method}_case6_tol_${tol}_dt_${dt}_tf_${tf}_grid_${nbsolpts}x${nb_elements}x${nb_elements}_using_${core}_cores
                        echo $testoutputFile

			# adjust configuration file code for kiops vs exode 
			if [[ $exode_method == "kiops" ]]
			then
				call_exode="time_integrator = $EPImethod"
			else
				call_exode="time_integrator = $EPImethod
exponential_solver=exode
exode_method = ${exode_method^^} 
exode_controller = PI3040 "
			fi

# create configuration file 
echo -e "
[General]

equations = shallow_water

[Grid]

# Possible values: cubed_sphere, cartesian2d
grid_type = cubed_sphere
λ0 = 0.0
#ϕ0 = 0.0
ϕ0 = 0.7853981633974483
α0 = 0.0

[Test_case]

# Possible values
#  -1 : multiscale signal (passive advection only)
#   0 : deformation flow  (passive advection only)
#   1 : cosine hill (passive advection only)
#   2 : zonal flow (shallow water)
#   5 : zonal flow over an isolated mountain (shallow water)
#   6 : Rossby-Haurvitz waves (shallow water)
#   8 : Unstable jet (shallow water)
case_number = 6

[Time_integration]

# Time step
dt = $dt  
#1800

# End time of the simulation in sec
t_end = $tf 
#1209600

# Time integration scheme
# Possible values  = tvdrk3 : 3th Order TVD-RK time integration
#                    epi2 : 2n order exponential propagation iterative
#                    epi3 : 3rd order exponential propagation iterative (Recommended)
#                    epirk4s3A : 4th order 3-stage EPIRK time integration
$call_exode

# Solver tolerance
tolerance = $tol

gmres_restart = 20

starting_step = 0


[Preconditioning]
preconditioner = none
precond_tolerance = 1e-1
num_mg_levels = 3
num_pre_smoothe = 1
num_post_smoothe = 1
pseudo_cfl = 2.5e7
restrict_method = modal
kiops_dt_factor = 1.2
mg_solve_coarsest = 0
mg_smoother = erk1
precond_filter_apply = 0
verbose_precond = 1
exp_smoothe_spectral_radii = [3.0, 1.0, 0.5, 2.2, 1.0]

[Spatial_discretization]

# The grid will have (nbsolpts) x (nbsolpts) nodal points in each elements.
nbsolpts = $nbsolpts

# Number of element in x^1 and x^2 directions
# Each face of the cube have (nbElements x nbElements) elements, for a total of (6 x nbElements x nbElements) elements.
nb_elements_horizontal = $nb_elements

[Output_options]

# Print blockstats every \"stat_freq\" steps, 0 to disable.
stat_freq = 0

# Output solution every \"output_freq\" steps, 0 to disable.
output_freq = $output_freq

# Save the state vector to a file at every \"save_state_freq\" steps. 0 to disable.
save_state_freq = 0

# Store statistics about the solver (iterations, residuals, etc.). 0 to disable.
store_solver_stats = 0

store_total_time = 1

# Output directory
output_dir  = ${testoutputSubDir}


        " > $configDir/$configFile
        begin=$(date +%s)	
        mpirun -n $core ./WxFactory ./config/${configFile} > ${testoutputSubDir}/${testoutputFile}.txt
	end=$(date +%s)
	echo "Elapsed Time: $(($end-$begin)) seconds"

done
done
done
done
done
done 
end_time=$(date +%s)

echo "Test completed!"
echo "Total time: $((end_time - start_time)) s"

