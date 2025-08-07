import sys
from time import time
from typing import List, Dict, Type

from mpi4py import MPI
import numpy

from common import Configuration
from common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w
from device import Device, CpuDevice, CudaDevice
from geometry import Cartesian2D, CubedSphere, CubedSphere3D, CubedSphere2D, DFROperators, Geometry
from init.dcmip import dcmip_T11_update_winds, dcmip_T12_update_winds
from init.init_state_vars import init_state_vars
from integrators import (
    Integrator,
    Epi,
    EpiStiff,
    Euler1,
    Imex2,
    PartRosExp2,
    Ros2,
    RosExp2,
    StrangSplitting,
    Srerk,
    Tvdrk3,
    BackwardEuler,
    CrankNicolson,
    Bdf2,
)
from output.output_manager import OutputManager
from output.output_cartesian import OutputCartesian
from output.output_cubesphere_netcdf import OutputCubesphereNetcdf
from output.output_cubesphere_fst import OutputCubesphereFst
from output.input_manager import InputManager
from precondition.factorization import Factorization
from precondition.multigrid import Multigrid
from process_topology import ProcessTopology
from rhs.rhs_selector import RhsBundle
from wx_mpi import SingleProcess, Conditional
from post_proccessing import PostProcessor, ScharMountainPostProcessor

class Simulation:
    """Encapsulate parameters and structures needed to run a WxFactory simulation.

    An object of this class is instantiated based on the content of a given config file. The
    config is read and validated, then all required structures are initialized (geometry, metric terms,
    initial state, etc.)

    Once the object is created, it can be used to step through the given problem one at a time, or to run
    it entirely.
    """

    config: Configuration
    post_processors: Dict[Type, PostProcessor]

    def __init__(
        self,
        config: Configuration | str,
        comm: MPI.Comm = MPI.COMM_WORLD,
        print_allowed_pe_counts: bool = False,
        quiet=False,
    ) -> None:
        """Create a Simulation object from a certain configuration.

        :type config: Configuration | str
        :param config: All options relevant to the simulation. Can be an already-initialized Configuration object, or
                       the name of a file where to find these options.
        """
        self.comm = comm
        self.rank = self.comm.rank

        self.post_processors = {}

        # self.input_manager = InputManager(self.comm)

        if isinstance(config, Configuration):
            self.config = config
        elif isinstance(config, str):
            self.config = InputManager.read_config(config, self.comm)
        else:
            raise ValueError(
                f"Need to provide either a Configuration or a config file name to create a Simulation\n"
                f"(Gave a {type(config)})"
            )

        if self.rank == 0 and not quiet:
            print(f"{self.config}", flush=True)

        if self.config.grid_file != "":
            self.num_elements_horizontal, self.num_solpts, self.lambda0, self.phi0, self.alpha0 = (
                InputManager.read_grid_params(self.config.grid_file, self.comm)
            )
        else:
            self.num_elements_horizontal = self.config.num_elements_horizontal
            self.num_solpts = self.config.num_solpts
            if self.config.grid_type == "cubed_sphere":
                self.lambda0 = self.config.lambda0
                self.phi0 = self.config.phi0
                self.alpha0 = self.config.alpha0

        self.allowed_pe_counts = (
            [
                i**2 * 6
                for i in range(1, max(self.num_elements_horizontal // 2 + 1, 2))
                if (self.num_elements_horizontal % i) == 0
            ]
            if self.config.grid_file != "" or self.config.grid_type == "cubed_sphere"
            else [1]
        )

        with SingleProcess(self.comm) as s, Conditional(s):
            if print_allowed_pe_counts:
                print(
                    f"Can use the following number of processes to run this configuration:\n"
                    f"  {self.allowed_pe_counts}"
                )
                raise SystemExit(0)

        self._adjust_num_elements()
        self.device = self._make_device()
        self.process_topo = None
        self.geometry = self._create_geometry()
        self.operators = DFROperators(self.geometry, self.config, self.device)
        self.initial_Q, self.topography, self.metric = init_state_vars(self.geometry, self.operators, self.config, self.post_processors)
        self.preconditioner = self._create_preconditioner(self.initial_Q)
        self.output = self._create_output_manager()
        self.initial_Q, self.starting_step = self._determine_starting_state()

        self.Q = self.initial_Q
        self.step_id = self.starting_step

        self.rhs = RhsBundle(
            self.geometry,
            self.operators,
            self.metric,
            self.topography,
            self.process_topo,
            self.config,
            self.initial_Q.shape,
            self.device,
        )

        self.integrator = self._create_time_integrator()
        self.integrator.output_manager = self.output
        self.integrator.device = self.device

        self.output.step(self.initial_Q, self.starting_step)
        sys.stdout.flush()

        self.t = self.config.dt * self.starting_step
        self.integrator.sim_time = self.t
        self.num_steps = int(numpy.ceil(self.config.t_end / self.config.dt)) - self.starting_step

    def step(self):
        """Advance the simulation by one time step."""
        if self.t < self.config.t_end:
            if self.t + self.config.dt > self.config.t_end:
                self.config.dt = self.config.t_end - self.t
                self.t = self.config.t_end
            else:
                self.t += self.config.dt

            self.step_id += 1

            if self.rank == 0:
                print(f"Step {self.step_id} of {self.num_steps + self.starting_step}", flush=True)

            self.Q = self.integrator.step(self.Q, self.config.dt)
            self.Q = self.operators.apply_filters(self.Q, self.geometry, self.metric, self.config.dt)

            if self.rank == 0:
                print(f"Elapsed time for step: {self.integrator.latest_time:.3f} secs", flush=True)

            # Check whether there are any NaNs in the solution
            # TODO put this inside the `step` function of the integrator
            self._check_for_nan(self.Q)

            # Overwrite winds for some DCMIP tests
            # TODO put this inside the `step` function of the integrator
            if self.config.case_number == 11:
                u1_contra, u2_contra, w_wind = dcmip_T11_update_winds(
                    self.geometry, self.metric, self.operators, self.config, time=self.t
                )
                self.Q[idx_rho_u1, :, :, :] = self.Q[idx_rho, :, :, :] * u1_contra
                self.Q[idx_rho_u2, :, :, :] = self.Q[idx_rho, :, :, :] * u2_contra
                self.Q[idx_rho_w, :, :, :] = self.Q[idx_rho, :, :, :] * w_wind
            elif self.config.case_number == 12:
                u1_contra, u2_contra, w_wind = dcmip_T12_update_winds(
                    self.geometry, self.metric, self.operators, self.config, time=self.t
                )
                self.Q[idx_rho_u1, :, :, :] = self.Q[idx_rho, :, :, :] * u1_contra
                self.Q[idx_rho_u2, :, :, :] = self.Q[idx_rho, :, :, :] * u2_contra
                self.Q[idx_rho_w, :, :, :] = self.Q[idx_rho, :, :, :] * w_wind

            for post_precessor_type in self.post_processors:
                self.post_processors[post_precessor_type].process()

            self.output.step(self.Q, self.step_id)  # Perform any requested output
            sys.stdout.flush()

            if self.integrator.failure_flag == 0:
                return True

        return False

    def run(self):
        """Run the entire simulation step by step"""
        self.step_id = self.starting_step
        self.Q = self.initial_Q

        start_time = time()

        while self.step():
            pass  # Step until everything is done

        self.output.finalize(time() - start_time)  # Close any open output file

    def _make_device(self) -> Device:
        """Create the device object which will determine on what hardware (CPU/GPU) each part of the simulation will
        be executed."""
        if self.config.desired_device in ["cuda", "cupy"]:
            try:
                cuda_devices = self.config.cuda_devices
            except AttributeError:
                cuda_devices = []
            try:
                device = CudaDevice(self.comm, cuda_devices)
            except ValueError:
                device = None
                if self.rank == 0:
                    print("Switching to CPU", flush=True)

            if device is None:
                device = CpuDevice(comm=self.comm)
        else:
            device = CpuDevice(comm=self.comm)

        return device

    def _adjust_num_elements(self):
        """Adjust number of horizontal elements in the parameters so that it corresponds to the
        number *per processor*."""
        if self.comm.size not in self.allowed_pe_counts:
            raise ValueError(
                f"Invalid number of processors for this particular "
                f"problem size ({self.num_elements_horizontal} elements per side). "
                f"\nAllowed counts are {self.allowed_pe_counts}"
            )

        self.total_num_elements_horizontal = self.num_elements_horizontal
        if self.comm.size > 1:
            num_pe_per_tile = self.comm.size // 6
            num_pe_per_line = int(numpy.sqrt(num_pe_per_tile))
            self.num_elements_horizontal = self.total_num_elements_horizontal // num_pe_per_line
            if self.rank == 0:
                if self.total_num_elements_horizontal != self.num_elements_horizontal:
                    print(
                        f"Adjusting horizontal number of elements from {self.total_num_elements_horizontal} "
                        f"(total) to {self.num_elements_horizontal} (per PE)"
                    )
                print(f"allowed_pe_counts = {self.allowed_pe_counts}", flush=True)

    def _create_geometry(self) -> Geometry:
        """Create the appropriate geometry for the given problem"""

        if self.config.grid_file != "":
            self.process_topo = ProcessTopology(self.device, comm_in=self.comm)
            return CubedSphere2D(
                self.num_elements_horizontal,
                self.num_solpts,
                self.total_num_elements_horizontal,
                self.lambda0,
                self.phi0,
                self.alpha0,
                self.process_topo,
            )

        if self.config.grid_type == "cubed_sphere":
            self.process_topo = ProcessTopology(self.device, comm_in=self.comm)
            if self.config.equations == "shallow_water":
                return CubedSphere2D(
                    self.num_elements_horizontal,
                    self.num_solpts,
                    self.total_num_elements_horizontal,
                    self.lambda0,
                    self.phi0,
                    self.alpha0,
                    self.process_topo,
                )
            elif self.config.equations == "euler":
                cube_sphere = CubedSphere3D(
                    self.num_elements_horizontal,
                    self.config.num_elements_vertical,
                    self.num_solpts,
                    self.total_num_elements_horizontal,
                    self.lambda0,
                    self.phi0,
                    self.alpha0,
                    self.config.ztop,
                    self.process_topo,
                    self.config,
                )
            
                if self.config.enable_schar_mountain:
                    schar_mountain = ScharMountainPostProcessor(self.config, cube_sphere)
                    self.post_processors[ScharMountainPostProcessor] = schar_mountain
                return cube_sphere

        if self.config.grid_type == "cartesian2d":
            return Cartesian2D(
                (self.config.x0, self.config.x1),
                (self.config.z0, self.config.z1),
                self.num_elements_horizontal,
                self.config.num_elements_vertical,
                self.num_solpts,
                self.total_num_elements_horizontal,
                self.device,
            )

        raise ValueError(f"Invalid grid type/process_topo: {self.config.grid_type}, {self.process_topo}")

    def _create_preconditioner(self, Q: numpy.ndarray) -> Multigrid | Factorization | None:
        """Create the preconditioner required by the given params"""
        if self.config.preconditioner != "none":
            raise ValueError(f"Preconditioner is currently unavalable, until it get fixed")

        if self.config.preconditioner == "p-mg":
            return Multigrid(self.config, self.process_topo, self.device, discretization="dg")
        if self.config.preconditioner == "fv-mg":
            return Multigrid(self.config, self.process_topo, self.device, discretization="fv")
        if self.config.preconditioner == "fv":
            return Multigrid(self.config, self.process_topo, self.device, discretization="fv", fv_only=True)
        if self.config.preconditioner in ["lu", "ilu"]:
            return Factorization(Q.dtype, Q.shape, self.config)
        return None

    def _create_output_manager(self) -> OutputManager:
        if isinstance(self.geometry, Cartesian2D):
            return OutputCartesian(self.config, self.geometry, self.operators, self.device)
        elif isinstance(self.geometry, CubedSphere):
            if self.config.output_format == "netcdf":
                return OutputCubesphereNetcdf(
                    self.config,
                    self.geometry,
                    self.operators,
                    self.device,
                    self.metric,
                    self.topography,
                    self.process_topo,
                )
            elif self.config.output_format == "fst":
                return OutputCubesphereFst(
                    self.config,
                    self.geometry,
                    self.operators,
                    self.device,
                    self.metric,
                    self.topography,
                    self.process_topo,
                )

        raise ValueError(f"Unrecognized geometry type {type(self.geometry)}")

    def _determine_starting_state(self):
        """Try to load the state for the given starting step and, if successful, swap it with the initial state"""
        if self.config.starting_step > 0:
            try:
                Q, starting_step = self.output.load_state_from_file(self.config.starting_step, self.initial_Q.shape)
                return Q, starting_step
            except (FileNotFoundError, ValueError, SystemExit) as e:
                if self.rank == 0:
                    print(
                        f"WARNING: Tried to start from timestep {self.config.starting_step}, but unable"
                        " to read initial state for that step. Will start from 0 instead."
                        f"\n{e}"
                    )
            except Exception as e:
                print(f"{self.rank} Fail with other ({type(e)})", flush=True)

        return self.initial_Q, 0

    def _create_time_integrator(self) -> Integrator:
        """Create the appropriate time integrator object based on params"""

        # --- Exponential time integrators
        if self.config.time_integrator[:9] == "epi_stiff" and self.config.time_integrator[9:].isdigit():
            order = int(self.config.time_integrator[9:])
            if self.comm.rank == 0:
                print(f"Running with EPI_stiff{order}")
            return EpiStiff(self.config, order, self.rhs.full, init_substeps=10, device=self.device)
        if self.config.time_integrator[:3] == "epi" and self.config.time_integrator[3:].isdigit():
            order = int(self.config.time_integrator[3:])
            if self.comm.rank == 0:
                print(f"Running with EPI{order}")
            return Epi(self.config, order, self.rhs.full, init_substeps=10, device=self.device)
        if self.config.time_integrator[:5] == "srerk" and self.config.time_integrator[5:].isdigit():
            order = int(self.config.time_integrator[5:])
            if self.comm.rank == 0:
                print(f"Running with SRERK{order}")
            return Srerk(self.config, order, self.rhs.full, device=self.device)

        # --- Explicit
        if self.config.time_integrator == "euler1":
            if self.comm.rank == 0:
                print("WARNING: Running with first-order explicit Euler timestepping.")
                print("         This is UNSTABLE and should be used only for debugging.")
            return Euler1(self.config, self.rhs.full, device=self.device)
        if self.config.time_integrator == "tvdrk3":
            return Tvdrk3(self.config, self.rhs.full, device=self.device)

        # --- Rosenbrock
        if self.config.time_integrator == "ros2":
            return Ros2(self.config, self.rhs.full, preconditioner=self.preconditioner, device=self.device)

        # --- Rosenbrock - Exponential
        if self.config.time_integrator == "rosexp2":
            return RosExp2(
                self.config, self.rhs.full, self.rhs.full, preconditioner=self.preconditioner, device=self.device
            )
        if self.config.time_integrator == "partrosexp2":
            return PartRosExp2(
                self.config, self.rhs.full, self.rhs.implicit, preconditioner=self.preconditioner, device=self.device
            )

        # --- Implicit - Explicit
        if self.config.time_integrator == "imex2":
            return Imex2(self.config, self.rhs.explicit, self.rhs.implicit, device=self.device)

        # --- Fully implicit
        if self.config.time_integrator == "backward_euler":
            return BackwardEuler(self.config, self.rhs.full, preconditioner=self.preconditioner, device=self.device)
        if self.config.time_integrator == "bdf2":
            return Bdf2(self.config, self.rhs.full, preconditioner=self.preconditioner, device=self.device)
        if self.config.time_integrator == "crank_nicolson":
            return CrankNicolson(self.config, self.rhs.full, preconditioner=self.preconditioner, device=self.device)

        # --- Operator splitting
        if self.config.time_integrator == "strang_epi2_ros2":
            stepper1 = Epi(self.config, 2, self.rhs.explicit, device=self.device)
            stepper2 = Ros2(self.config, self.rhs.implicit, preconditioner=self.preconditioner, device=self.device)
            return StrangSplitting(self.config, stepper1, stepper2)
        if self.config.time_integrator == "strang_ros2_epi2":
            stepper1 = Ros2(self.config, self.rhs.implicit, preconditioner=self.preconditioner, device=self.device)
            stepper2 = Epi(self.config, 2, self.rhs.explicit)
            return StrangSplitting(self.config, stepper1, stepper2)

        raise ValueError(f"Time integration method {self.config.time_integrator} not supported")

    def _check_for_nan(self, Q):
        """Raise an exception if there are NaNs in the input"""
        error_detected = numpy.array([0], dtype=numpy.int32)
        if numpy.any(numpy.isnan(Q)):
            print(f"NaN detected on process {self.comm.rank}")
            error_detected[0] = 1
        error_detected_out = numpy.zeros_like(error_detected)
        self.comm.Allreduce(error_detected, error_detected_out, MPI.MAX)
        if error_detected_out[0] > 0:
            raise ValueError(f"NaN")