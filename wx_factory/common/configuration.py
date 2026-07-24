from configparser import ConfigParser
import copy
from typing import Dict, List, Optional, Self
from .eval_expr import eval_expr

from .configuration_schema import ConfigurationSchema, ConfigurationField, OptionType, needs_evaluation

__all__ = ["Configuration"]


class Configuration:
    """All the config options for WxFactory"""

    sections: Dict[str, List[str]]

    def __init__(self, cfg_file: str, schema: ConfigurationSchema, load_post_config: bool = True):

        self.cfg_file = "in-memory"
        self.sections = {}
        self.schema = schema
        self.parser = ConfigParser()

        self.config_content = cfg_file

        self.depth_approx = None

        self.parser.read_string(self.config_content)

        for field in schema.fields:
            self._get_option(field)

        self.state_version = schema.version

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
                return None

            values = [
                (
                    eval_expr(v)
                    if needs_evaluation(field.dependency[1][0], type(getattr(self, field.dependency[0])))
                    else v
                )
                for v in field.dependency[1]
            ]
            if not getattr(self, field.dependency[0]) in values:
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
                numeric = getattr(self, option)
                if isinstance(numeric, float):
                    val = f"{numeric:.6g}"
                else:
                    val = str(numeric)

                if len(option) < 26 and len(val) < 14:
                    if i % 2 == 0:
                        out += "\n"
                    out += f" | {option:25s}: {val:13s}"
                    i += 1
                else:
                    long_options[option] = val
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
    cuda_devices: List[int]
    depth_approx: str
    desired_device: str
    discretization: str
    dt: float
    enable_schar_mountain: bool
    equations: str
    exode_controller: str
    exode_method: str
    expfilter_apply: bool
    expfilter_cutoff: float
    expfilter_order: int
    expfilter_strength: float
    exponential_solver: str
    filter_apply: bool
    filter_cutoff: float
    filter_order: int
    gmres_restart: int
    grid_file: str
    grid_type: str
    initial_condition: str
    initial_conditions_file: str
    jacobian_method: str
    kiops_dt_factor: float
    krylov_mmax: int
    krylov_size: int
    lambda0: float
    matmul_backend: str
    matsuno_amp: float
    matsuno_wave_type: str
    num_elements_horizontal: int
    num_elements_vertical: int
    num_solpts: int
    output_dir: str
    output_format: str
    output_freq: int
    phi0: float
    precision: str
    preconditioner: str
    pytorch_device: str
    save_state_freq: int
    schar_mountain_height: float
    schar_mountain_lattitude: float
    schar_mountain_length: float
    schar_mountain_longitude: float
    schar_mountain_radius: float
    schar_mountain_step: int
    sleve_scale_large: float
    sleve_scale_small: float
    splitting_integrator_1: str
    splitting_integrator_2: str
    sponge_tscale: float
    sponge_zscale: float
    starting_step: int
    stat_freq: int
    store_total_time: bool
    t_end: float
    time_end: str
    time_integrator: str
    time_start: str
    tolerance: float
    topography_file: str
    verbose_precond: int
    verbose_solver: int
    vertical_coord: str
    x0: float
    x1: float
    z0: float
    z1: float
    ztop: float
    # --- END type hints --- automatically generated (do not touch)
