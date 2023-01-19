import numpy
import math
import mpi4py.MPI

from Output.diagnostic import total_energy, potential_enstrophy, global_integral

def blockstats(Q, geom, topo, metric, mtrx, param, step):

   h  = Q[0,:,:]

   if param.case_number == 0:
      h_anal, _ = height_vortex(geom, metric, param, step)
   elif param.case_number == 1:
      h_anal = height_case1(geom, metric, param, step)
   elif param.case_number == 2:
      h_anal = height_case2(geom, metric, param)
   elif param.case_number == 10:
      h_anal = height_unsteady_zonal(geom, metric, param)

   if param.case_number > 1:
      u1_contra = Q[1,:,:] / h
      u2_contra = Q[2,:,:] / h

   if param.case_number >= 2:
      energy = total_energy(h, u1_contra, u2_contra, geom, topo, metric)
      enstrophy = potential_enstrophy(h, u1_contra, u2_contra, geom, metric, mtrx, param)

   print("\n================================================================================================")

   if step == 0:
      print("Blockstats for initial conditions")

      if param.case_number >= 2:
         global initial_mass
         global initial_energy
         global initial_enstrophy
         initial_mass = global_integral(h, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
         initial_energy = global_integral(energy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
         initial_enstrophy = global_integral(enstrophy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 

         print(f'Integral of mass = {initial_mass}')
         print(f'Integral of energy = {initial_energy}')
         print(f'Integral of enstrophy = {initial_enstrophy}')

   else:
      print("Blockstats for timestep ", step)

      if param.case_number <= 2 or param.case_number == 10:
         absol_err = global_integral(abs(h - h_anal), mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
         int_h_anal = global_integral(abs(h_anal), mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
   
         absol_err2 = global_integral((h - h_anal)**2, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
         int_h_anal2 = global_integral(h_anal**2, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
   
         max_absol_err = mpi4py.MPI.COMM_WORLD.allreduce(numpy.max(abs(h - h_anal)), op=mpi4py.MPI.MAX)
         max_h_anal = mpi4py.MPI.COMM_WORLD.allreduce(numpy.max(h_anal), op=mpi4py.MPI.MAX)
   
         l1 = absol_err / int_h_anal
         l2 = math.sqrt( absol_err2 / int_h_anal2 )
         linf = max_absol_err / max_h_anal
         print(f'l1 = {l1} \t l2 = {l2} \t linf = {linf}')
      
      if param.case_number >= 2:
            int_mass = global_integral(h, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
            int_energy = global_integral(energy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
            int_enstrophy = global_integral(enstrophy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal) 
         
            normalized_mass = ( int_mass - initial_mass ) / initial_mass
            normalized_energy = ( int_energy - initial_energy ) / initial_energy
            normalized_enstrophy = ( int_enstrophy - initial_enstrophy ) / initial_enstrophy
            print(f'normalized integral of mass = {normalized_mass}')
            print(f'normalized integral of energy = {normalized_energy}')
            print(f'normalized integral of enstrophy = {normalized_enstrophy}')


   print("================================================================================================")
