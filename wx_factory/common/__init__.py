import os

from .configuration import Configuration
from .configuration_schema import ConfigurationSchema, load_default_schema

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
main_module_dir = os.path.join(main_project_dir, "wx_factory")

__all__ = ["Configuration", "ConfigurationSchema", "load_default_schema", "main_project_dir", "main_module_dir"]
