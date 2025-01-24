from configparser import ConfigParser
import copy
from typing import Dict, List, Optional, Self

from .configuration_schema import ConfigurationSchema, ConfigurationField, OptionType

__all__ = ["Configuration"]


class Configuration:
    """All the config options for GEF"""

    sections: Dict[str, List[str]]

    def __init__(self, cfg_file: str, schema: ConfigurationSchema, load_post_config: bool = True):

        self.cfg_file = "in-memory"
        self.sections = {}
        self.parser = ConfigParser()

        self.config_content = cfg_file

        self.depth_approx = None
        self.num_mg_levels = 1

        self.parser.read_string(self.config_content)

        for field in [field for field in schema.fields if field.dependency is None]:
            try:
                self._get_option(field)
            except Exception as e:
                raise ValueError(f"Error reading option {field.name}") from e

        for field in [field for field in schema.fields if field.dependency is not None]:
            try:
                self._get_option(field)
            except Exception as e:
                raise ValueError(f"Error reading option {field.name}") from e

        self.state_version = schema.version

        if load_post_config:
            self.initial_num_solpts = self.num_solpts
            self.num_elements_horizontal_total = self.num_elements_horizontal

            if self.discretization == "fv":
                if self.num_solpts != 1:
                    raise ValueError(
                        f"The number of solution of solution points ({self.num_solpts}) in configuration file"
                        " is inconsistent with a finite volume discretization"
                    )

            if self.mg_smoother == "exp":
                self.exp_smoothe_spectral_radius = self.exp_smoothe_spectral_radii[0]
                self.exp_smoothe_num_iter = self.exp_smoothe_num_iters[0]

    def __deepcopy__(self: Self, memo) -> Self:
        do_not_deepcopy = {}
        other = copy.copy(self)
        for k, v in vars(self).items():
            if k not in do_not_deepcopy:
                setattr(other, k, copy.deepcopy(v, memo))
        return other

    def _get_option(self, field: ConfigurationField) -> OptionType:
        value: Optional[OptionType] = None
        if field.dependency is not None:
            if not hasattr(self, field.dependency[0]):
                raise ValueError(f"Cannot validate dependency {field.dependency[0]}. dependency not found")

            if not getattr(self, field.dependency[0]) in field.dependency[1]:
                return None

        value = field.read(self.parser)
        setattr(self, field.name, value)

        if field.section not in self.sections:
            self.sections[field.section] = []
        self.sections[field.section].append(field.name)

        return value

    def __str__(self):
        out = "Configuration: \n"
        for section_name, section_options in self.sections.items():
            out += "\n"
            out += f'  {" " + section_name + " ":-^80s}  '
            long_options = {}
            i = 0
            for option in section_options:
                val = str(getattr(self, option))

                if len(option) < 26 and len(val) < 14:
                    if i % 2 == 0:
                        out += "\n"
                    out += f" | {option:25s}: {val:13s}"
                    i += 1
                else:
                    long_options[option] = val
                    # i = 1
            if i % 2 == 1:
                out += " |"

            for name, val in long_options.items():
                out += f"\n | {name:25s}: {val}"
            out += "\n"

        return out

    # --- START type hints --- automatically generated (do not touch)
    alpha0: float
    apply_sponge: bool
    base_output_file: str
    bubble_rad: float
    bubble_theta: float
    case_number: int
    cuda_devices: List
    depth_approx: str
    desired_device: str
    dg_to_fv_interp: str
    discretization: str
    dt: float
    equations: str
    exode_controller: str
    exode_method: str
    expfilter_apply: bool
    expfilter_cutoff: float
    expfilter_cutoff: float
    expfilter_order: int
    expfilter_order: int
    expfilter_strength: float
    expfilter_strength: float
    exponential_solver: str
    exp_smoothe_num_iters: List
    exp_smoothe_spectral_radii: List
    filter_apply: bool
    filter_cutoff: float
    filter_cutoff: float
    filter_order: int
    filter_order: int
    gmres_restart: int
    grid_type: str
    jacobian_method: str
    kiops_dt_factor: float
    krylov_size: int
    lambda0: float
    linear_solver: str
    matsuno_amp: float
    matsuno_wave_type: str
    mg_smoother: str
    mg_solve_coarsest: bool
    num_elements_horizontal: int
    num_elements_vertical: int
    num_mg_levels: int
    num_post_smoothe: int
    num_pre_smoothe: int
    num_solpts: int
    output_dir: str
    output_format: str
    output_freq: int
    phi0: float
    precond_flux: str
    preconditioner: str
    precond_tolerance: float
    pseudo_cfl: float
    save_state_freq: int
    solver_stats_file: str
    sponge_tscale: float
    sponge_zscale: float
    starting_step: int
    stat_freq: int
    store_solver_stats: bool
    store_total_time: bool
    t_end: float
    time_integrator: str
    tolerance: float
    verbose_precond: int
    verbose_solver: int
    x0: float
    x1: float
    z0: float
    z1: float
    ztop: float
    # --- END type hints --- automatically generated (do not touch)
