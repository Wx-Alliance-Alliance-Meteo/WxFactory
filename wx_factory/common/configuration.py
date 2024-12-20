from configparser import ConfigParser, NoSectionError, NoOptionError
import copy
import json
from typing import Dict, List, Optional, Type, Self

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

        for field in [field for field in schema.fields if field.dependancy is None]:
            self._get_option(field)

        for field in [field for field in schema.fields if field.dependancy is not None]:
            self._get_option(field)

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

            self.output_file = f"{self.output_dir}/{self.base_output_file}.nc"

    def __deepcopy__(self: Self, memo) -> Self:
        do_not_deepcopy = {}
        other = copy.copy(self)
        for k, v in vars(self).items():
            if k not in do_not_deepcopy:
                setattr(other, k, copy.deepcopy(v, memo))
        return other

    def _get_opt_from_parser(self, section_name: str, option_name: str, option_type: Type[OptionType]) -> OptionType:
        value: Optional[OptionType] = None

        if option_type == float:
            value = self.parser.getfloat(section_name, option_name)
        elif option_type == int:
            value = self.parser.getint(section_name, option_name)
        elif option_type == str:
            value = self.parser.get(section_name, option_name)
        elif option_type == bool:
            value = self.parser.getint(section_name, option_name) > 0
        elif option_type == List[int]:
            try:
                value = [self.parser.getint(section_name, option_name)]
            except ValueError:
                value = [int(x) for x in json.loads(self.parser.get(section_name, option_name))]
        elif option_type == List[float]:
            try:
                value = [self.parser.getfloat(section_name, option_name)]
            except ValueError:
                value = [float(x) for x in json.loads(self.parser.get(section_name, option_name))]
        else:
            raise ValueError(f"Cannot get this option type (not implemented): {option_type}")

        assert value is not None
        return value

    def _get_option(self, field: ConfigurationField) -> OptionType:
        value: Optional[OptionType] = None
        if field.dependancy is not None:
            if not hasattr(self, field.dependancy[0]):
                raise ValueError(f"Cannot validate depancy {field.dependancy[0]}. Dependancy not found")

            if not getattr(self, field.dependancy[0]) in field.dependancy[1]:
                return None

        try:
            value = field.transform(self._get_opt_from_parser(field.field_section, field.field_name, field.field_type))

            if not field.validate(value):
                raise ValueError(f"Cannot validate {field.field_name} at section {field.field_section}")
        except (NoOptionError, NoSectionError) as e:
            if field.field_default is None:
                e.message += f"\nMust specify a value for option '{field.field_name}' in file {self.cfg_file}"
                raise
            value = field.field_default

        setattr(self, field.field_name, value)

        if field.field_section not in self.sections:
            self.sections[field.field_section] = []
        self.sections[field.field_section].append(field.field_name)

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
