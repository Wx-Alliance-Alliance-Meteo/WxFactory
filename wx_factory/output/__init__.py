from .input_manager import InputManager
from .state import load_state, save_state

# NOTE: the output registry (registry.py) is intentionally not imported here. It pulls in the
# cubed-sphere output managers, which import wx_factory.init.shallow_water, which in turn imports
# this package -- importing the registry from __init__ would close that cycle. Import it directly
# as `from wx_factory.output.registry import resolve_output` instead.

__all__ = ["InputManager", "load_state", "save_state"]
