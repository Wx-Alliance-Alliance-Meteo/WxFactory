.. WxFactory documentation master file, created by
   sphinx-quickstart on Thu Jul 11 20:12:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Welcome to WxFactory's documentation!
.. =====================================

.. include:: ../../../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   doc/config_options
   doc/contribute
   doc/references
   tests/readme


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


API
-------------------
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   wx_factory.common
   wx_factory.device
   wx_factory.geometry
   wx_factory.init
   wx_factory.integrators
   wx_factory.output
   wx_factory.pde
   wx_factory.precondition
   wx_factory.process_topology
   wx_factory.rhs
   wx_factory.simulation
   wx_factory.solvers
   wx_factory.step_hooks
   wx_factory.wx_mpi
