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

   doc/config_options.md
   doc/contribute.md
   doc/references.md
   tests/readme.md


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

   common
   compiler
   geometry
   init
   integrators
   output
   precondition
   rhs
   solvers
   tests
   wx_mpi
