import sys
from time import time

import numpy
import torch
from mpi4py import MPI

from ..common import Configuration
from ..context import Context
from ..geometry import DFROperators, GeometryContext, Metric3DTopo, resolve_geometry
from ..geometry.geometry import cast_double_arrays
from ..init.init_state_vars import init_state_vars
from ..integrators import Integrator, resolve as _resolve_integrator
from ..output.input_manager import InputManager
from ..output.registry import OutputContext, resolve_output
from ..precondition import PreconditionerContext, resolve_preconditioner
from ..rhs.rhs_selector import RhsContext, resolve_rhs
from ..step_hooks import StepHook
from ..step_hooks.registry import (
    PHASE_GEOMETRY,
    PHASE_STATE,
    StepHookContext,
    resolve_step_hooks,
)
from ..wx_mpi import Conditional, SingleProcess


class Simulation:
    """Encapsulate parameters and structures needed to run a WxFactory simulation.

    An object of this class is instantiated based on the content of a given config file. The
    config is read and validated, then all required structures are initialized (geometry, metric terms,
    initial state, etc.)

    Once the object is created, it can be used to step through the given problem one at a time, or to run
    it entirely.
    """

    config: Configuration
    step_hooks: dict[type, StepHook]

    def __init__(
        self,
        config: Configuration | str,
        comm: MPI.Comm = MPI.COMM_WORLD,
        print_allowed_pe_counts: bool = False,
        quiet: bool = False,
        context: Context | None = None,
    ) -> None:
        """Create a Simulation object from a certain configuration.

        :type config: Configuration | str
        :param config: All options relevant to the simulation. Can be an already-initialized Configuration object, or
                       the name of a file where to find these options.
        """
        self.comm = comm
        self.rank = self.comm.rank

        self.step_hooks = {}

        if isinstance(config, Configuration):
            self.config = config
        elif isinstance(config, str):
            self.config = InputManager.read_config(config, self.comm)
        else:
            raise TypeError(
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
                    f"Can use the following number of processes to run this configuration:\n  {self.allowed_pe_counts}"
                )
                raise SystemExit(0)

        self._adjust_num_elements()
        self.context = self._make_context(context)

        # Mixed mode stores the model state and most runtime arrays in float32. Static spatial
        # coefficients are constructed in float64 before casting, and accuracy-sensitive solver
        # operations selectively retain or accumulate in float64.
        runtime_dtype = torch.float32 if self.config.precision == "mixed" else torch.float64

        # Build the geometry, the metric terms and the initial state in double precision, then store
        # them once in the working precision.
        self.context.real_dtype = torch.float64

        self.geometry = resolve_geometry(GeometryContext.from_simulation(self))
        # Cubed-sphere geometries carry a process topology; a Cartesian grid has none.
        self.process_topo = getattr(self.geometry, "process_topology", None)
        # Geometry-phase step hooks must exist before init_state_vars, which reads them.
        self.step_hooks.update(
            resolve_step_hooks(StepHookContext(config=self.config, geometry=self.geometry), phase=PHASE_GEOMETRY)
        )
        operators_double = DFROperators(self.geometry, self.context)
        self.initial_state = init_state_vars(self.geometry, operators_double, self.config, self.step_hooks)
        # Preserve the pre-restart double state for logarithmic extrapolation.
        q_ref_double = self.initial_state.Q.clone()

        self.context.real_dtype = runtime_dtype
        if runtime_dtype != torch.float64:
            self.geometry.dtype = runtime_dtype
            cast_double_arrays(self.geometry, runtime_dtype)
            cast_double_arrays(self.initial_state.metric, runtime_dtype)
            if self.initial_state.topography is not None:
                cast_double_arrays(self.initial_state.topography, runtime_dtype)
            self.initial_state.Q = self.initial_state.Q.to(runtime_dtype)

        # Rebuilt against the stored geometry so the runtime operators carry the working precision.
        self.operators_real = DFROperators(self.geometry, self.context)
        if isinstance(self.initial_state.metric, Metric3DTopo):
            # Terrain ramping rebuilds this metric in the runtime precision.
            self.initial_state.metric.matrix = self.operators_real

        self.output = resolve_output(
            OutputContext(
                config=self.config,
                context=self.context,
                geometry=self.geometry,
                operators=self.operators_real,
                metric=self.initial_state.metric,
                topography=self.initial_state.topography,
                ptopo=self.process_topo,
            )
        )
        self.initial_state.Q, self.starting_step = self._determine_starting_state()

        self.Q = self.initial_state.Q.clone()
        self.step_id = self.starting_step

        self.rhs = resolve_rhs(
            RhsContext(
                geom=self.geometry,
                operators_real=self.operators_real,
                metric=self.initial_state.metric,
                topo=self.initial_state.topography,
                ptopo=self.process_topo,
                param=self.config,
                fields_shape=self.initial_state.Q.shape,
                q_ref=q_ref_double,
                operators_double=operators_double,
            )
        )

        self.preconditioner = resolve_preconditioner(
            PreconditionerContext(
                config=self.config,
                context=self.context,
                geometry=self.geometry,
                operators=self.operators_real,
                rhs=self.rhs,
                metric=self.initial_state.metric,
                topography=self.initial_state.topography,
                ptopo=self.process_topo,
                fields_shape=self.initial_state.Q.shape,
            )
        )

        # State-phase step hooks can now be built (they need the metric and operators).
        self.step_hooks.update(
            resolve_step_hooks(
                StepHookContext(
                    config=self.config,
                    geometry=self.geometry,
                    operators=self.operators_real,
                    metric=self.initial_state.metric,
                ),
                phase=PHASE_STATE,
            )
        )

        self.integrator = self._create_time_integrator(self.config.time_integrator)
        self.integrator.output_manager = self.output
        self.integrator.context = self.context

        self.output.step(self.initial_state.Q, self.starting_step)
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

            if self.rank == 0:
                print(f"Elapsed time for step: {self.integrator.latest_time:.3f} secs", flush=True)

            # Check whether there are any NaNs in the solution
            # TODO put this inside the `step` function of the integrator
            self._check_for_nan(self.Q)

            for hook_type in self.step_hooks:
                self.Q = self.step_hooks[hook_type].process(self.Q, self.t)

            self.output.step(self.Q, self.step_id)  # Perform any requested output
            sys.stdout.flush()

            if self.integrator.failure_flag == 0:
                return True

        return False

    def run(self):
        """Run the entire simulation step by step"""
        self.step_id = self.starting_step
        self.Q = self.initial_state.Q

        start_time = time()

        while self.step():
            pass  # Step until everything is done

        # print_times is a diagnostic of the instrumented DFR right-hand side; RhsAdvection2d has no
        # timing instrumentation and does not provide it.
        if self.rank == 0 and hasattr(self.rhs.full, "print_times"):
            self.rhs.full.print_times()

        self.output.finalize(time() - start_time)  # Close any open output file

    def _make_context(self, context: Context | None) -> Context:
        """Create the context object which will determine on what hardware (CPU/GPU) each part of the simulation will
        be executed."""
        if context is not None:
            self.comm = context.comm
        else:
            context = Context(comm=self.comm, device_type=self.config.pytorch_device)
        # Share this context with helpers and nested integrators.
        Context.set_default(context)
        return context

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

    def _determine_starting_state(self):
        """Try to load the state for the given starting step and, if successful, swap it with the initial state"""
        if self.config.starting_step > 0:
            try:
                Q, starting_step = self.output.load_state_from_file(
                    self.config.starting_step, self.initial_state.Q.shape
                )
                return Q, starting_step
            except (FileNotFoundError, ValueError, SystemExit) as e:
                if self.rank == 0:
                    print(
                        f"WARNING: Tried to start from timestep {self.config.starting_step}, but unable"
                        " to read initial state for that step. Will start from 0 instead."
                        f"\n{e}"
                    )
            # Restart loading crosses file, NumPy, backend, and MPI boundaries. Any failure must
            # fall back consistently on every rank rather than leave some ranks inside a collective.
            except Exception as e:  # noqa: BLE001
                print(f"{self.rank} Fail with other ({type(e)})", flush=True)

        return self.initial_state.Q, 0

    def _create_time_integrator(self, name: str) -> Integrator:
        """Create the appropriate time integrator object based on params"""
        if self.comm.rank == 0:
            print(f"Running with time integrator: {name}")
        return _resolve_integrator(name, self.config, self.rhs, self.preconditioner, self.context)

    def _check_for_nan(self, Q):
        """Raise an exception if there are NaNs in the input"""
        # Reduce on-device so we only transfer a single scalar to the host, and avoid
        # a full-state D2H PCIe copy of Q just to scan it for NaNs.
        has_nan = bool(torch.isnan(Q).any().item())
        error_detected = numpy.array([0], dtype=numpy.int32)
        if has_nan:
            print(f"NaN detected on process {self.comm.rank}")
            error_detected[0] = 1
        error_detected_out = numpy.zeros_like(error_detected)
        self.comm.Allreduce(error_detected, error_detected_out, MPI.MAX)
        if error_detected_out[0] > 0:
            raise ValueError("NaN")
