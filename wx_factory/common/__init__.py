import os

from . import angle24, config_hints
from .configuration import Configuration
from .configuration_schema import ConfigurationSchema, ConfigValueError, default_schema_path, load_default_schema
from .grid_encoding import decode_ig4, make_ig4
from .readfile import readfile

main_project_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
main_module_dir = os.path.join(main_project_dir, "wx_factory")


__all__ = [
    "ConfigValueError",
    "Configuration",
    "ConfigurationSchema",
    "angle24",
    "config_hints",
    "decode_ig4",
    "default_schema_path",
    "load_default_schema",
    "main_module_dir",
    "main_project_dir",
    "make_ig4",
    "readfile",
]
