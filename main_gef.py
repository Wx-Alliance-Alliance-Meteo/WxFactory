#!/usr/bin/env python3

""" The GEF model """

import sys
import traceback

from mpi4py import MPI
import numpy

from common.array_module import ArrayModule

# array is the module used by GEF for managing arrays when the type is only known at runtime.
# This is so that we can use either CUDA or numpy arrays, depending on what the user has requested
# and what is available. By default, we use numpy arrays
array: ArrayModule = numpy

if __name__ == '__main__':

   args = None
   rank = MPI.COMM_WORLD.rank

   try:
      import argparse
      import cProfile
      import os.path

      from common.program_options import Configuration

      parser = argparse.ArgumentParser(description='Solve NWP problems with GEF!')
      parser.add_argument('--profile', action='store_true', help='Produce an execution profile when running')
      parser.add_argument('config', type=str, help='File that contains simulation parameters')
      parser.add_argument('--show-every-crash', action='store_true',
                          help='In case of an exception, show output from alllllll PEs')
      parser.add_argument('--numpy-warn-as-except', action='store_true',
                          help='Raise an exception if there is a numpy warning')

      args = parser.parse_args()

      if not os.path.exists(args.config):
         raise ValueError(f'Config file does not seem valid: {args.config}')

      # Start profiling
      pr = None
      if args.profile:
         pr = cProfile.Profile()
         pr.enable()

      numpy.set_printoptions(suppress=True, linewidth=256)

      if args.numpy_warn_as_except:
         numpy.seterr(all='raise')

      # Read configuration file
      config = Configuration(args.config, rank == 0)

      # Run the actual GEF
      # This import has to happen *after* the config is read, so that we have to swap the array
      # module for cupy if requested by the user (and available)
      import run
      run.run(config)

      if args.profile and pr:
         pr.disable()

         out_file = f'prof_{rank:04d}.out'
         pr.dump_stats(out_file)

   except Exception as e:

      sys.stdout.flush()
      if args and args.show_every_crash:
         traceback.print_exc()
      else:
         if rank == 0:
            traceback.print_exc()
            print(f'There was an error while running GEF. Only rank 0 is printing the traceback.')

      sys.exit(-1)
