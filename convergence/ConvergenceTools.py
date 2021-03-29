import math
from time import time

import mpi4py.MPI
import numpy
import json
import hashlib
import os
import time

from blockstats      import blockstats
from cubed_sphere    import cubed_sphere
from initialize      import initialize_sw, initialize_euler
from matrices        import DFR_operators
from metric          import Metric
from output          import output_init, output_netcdf, output_finalize
from parallel        import Distributed_World
from program_options import Configuration
from rhs_sw          import rhs_sw
from rhs_sw_explicit import rhs_sw_explicit
from rhs_sw_implicit import rhs_sw_implicit
from timeIntegrators import Epi, Epirk4s3a, Tvdrk3, Rat2, ARK_epi2

class ConvergenceAnalyzer:

    rhs = None  # ODE RHS
    rhs_explicit = None # Explicit ODE RHS for IMEX
    rhs_implicit = None # Implicit ODE RHS for IMEX
    Q0  = None  # ODE initial condition
    params = None  # Problem parameters
    _reference_integrator = None
    _reference_solution = None
    _verbose = True
    output_directory = None

    def __init__(self, args, verbose = True, output_directory = "output/reference"):
        self.rhs, self.Q0, self.rhs_explicit, self.rhs_implicit = self.initProblem(args)
        
        self._reference_integrator = Epi(4, self.rhs, self.params.tolerance, self.params.krylov_size, init_substeps=10)
        
        self._verbose = self._verbose
        self.output_directory = output_directory

    def initProblem(self, args):
        
        # Read configuration file
        param = Configuration(args.config)
        self.params = param

        # Set up distributed world
        ptopo = Distributed_World()

        # Create the mesh
        geom = cubed_sphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0, param.α0, param.ztop, ptopo)

        # Build differentiation matrice and boundary correction
        mtrx = DFR_operators(geom, param)

        # Initialize metric tensor
        metric = Metric(geom)

        # Initialize state variables
        if param.equations == "shallow water":
            Q, topo = initialize_sw(geom, metric, mtrx, param)
        elif param.equations == "Euler":
            Q, topo = initialize_euler(geom, metric, mtrx, param)

        # Initailize ODE RHS
        rhs_handle = lambda q: rhs_sw(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal, param.case_number, param.filter_apply)
        rhs_explicit = lambda q: rhs_sw_explicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal, param.case_number, param.filter_apply)
        rhs_implicit = lambda q: rhs_sw_implicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal, param.case_number, param.filter_apply)

        return rhs_handle, Q, rhs_explicit, rhs_implicit

    
    def analyze(self, output_filepath, integrators, Nts, Nt_reference=None):
        
        if( Nt_reference is None):
            Nt_reference = 2 * max(Nts)
        self.initReferenceSolution(Nt_reference)

        NtToH = lambda Nt : self.params.t_end / Nt
        output = { "experiment" : { "Nts" : Nts, "hs" : list(map(NtToH, Nts))}, "results" : {} }
        Nts_len = len(Nts)
        
        for i in range(0, len(integrators)):
            
            integrator = integrators[i]
            errors = [ None ] * Nts_len
            times  = [ None ] * Nts_len

            for j in range(0, Nts_len):
                Q, elapsed_time = self.solve(integrator["object"], Nts[j])
                
                times[j] = elapsed_time
                errors[j] = self.errorNorm(Q)                
                output["results"][integrator["name"]] = { "errors" : errors , "times" : times}

                self.saveOutput(output, output_filepath)

    def solve(self, integrator, Nt):

        dt = self.params.t_end / Nt
        Q  = numpy.copy(self.Q0)

        start_time = time.time()

        for i in range(0, Nt):
             Q = integrator.step(Q, dt)
             self.__vprint(f"\tstep {i}/{Nt}")
        
        elapsed_time = time.time() - start_time
        return Q, elapsed_time

    def errorNorm(self, Q):

        Q_full, rank = self.__gatherSolution(Q)
        if ( rank == 0 ):
            return numpy.linalg.norm( self._reference_solution - Q_full.flatten(), ord = numpy.inf )
    
    def saveOutput(self, data, filepath):
        
        rank  = mpi4py.MPI.COMM_WORLD.Get_rank()
        
        if ( rank == 0 ):
            with open(filepath, "w") as outfile:
                json.dump(data, outfile)

        mpi4py.MPI.COMM_WORLD.Barrier()

    def initReferenceSolution(self, Nt):

        if ( not os.path.exists(self.__referenceFilepath(Nt)) ):
            self.__computeReferenceSolution(Nt)
            self.__saveReferenceSolution(Nt)
        else:
            self.__loadReferenceSolution(Nt)

    def __computeReferenceSolution(self, Nt):
        self.__vprint("Computing Reference Solution: ")
        Q, *_ = self.solve(
            self._reference_integrator,
            Nt
        )
        Q_full, rank = self.__gatherSolution(Q)
        if ( rank == 0 ):
            self._reference_solution = Q_full.flatten()

    def __saveReferenceSolution(self, Nt):
        
        rank  = mpi4py.MPI.COMM_WORLD.Get_rank()

        if ( rank == 0 ):
            numpy.savez_compressed(
                self.__referenceFilepath(Nt),
                self._reference_solution
            )
        mpi4py.MPI.COMM_WORLD.Barrier()

    def __loadReferenceSolution(self, Nt):
                
        rank  = mpi4py.MPI.COMM_WORLD.Get_rank()
        
        if ( rank == 0 ):
            self.__vprint(f"Loading Reference Solution: {self.__referenceFilepath(Nt)}")
            data_raw = numpy.load(
                self.__referenceFilepath(Nt)
            )
            self._reference_solution = data_raw['arr_0']
        
        mpi4py.MPI.COMM_WORLD.Barrier()
    
    # gather reference solution on proc with rank 0
    def __gatherSolution(self, solution):
        
        size  = mpi4py.MPI.COMM_WORLD.Get_size()
        rank  = mpi4py.MPI.COMM_WORLD.Get_rank()
        
        shape = list(solution.shape)
        shape.insert(0, size)

        recvbuf = None
        if ( rank == 0 ):
            recvbuf = numpy.empty(shape)
        
        mpi4py.MPI.COMM_WORLD.Gather(solution, recvbuf, root=0)
        return recvbuf, rank
    
    def __referenceFilepath(self, Nt):
        m5d_str = hashlib.md5(json.dumps(self.params.__dict__).encode()).hexdigest()
        return f"{self.output_directory}/{m5d_str}-{Nt}.npz"

    def __vprint(self, message):
        if( self._verbose ):
            print(message)
