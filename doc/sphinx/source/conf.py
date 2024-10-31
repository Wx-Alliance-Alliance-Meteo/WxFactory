# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[3].resolve().as_posix())

# for dir in ['common', 'geometry', 'init', 'integrators', 'output', 'precondition', 'rhs', 'solvers', 'wx_cupy']:
#     sys.path.insert(0, os.path.abspath(f'../../../{dir}'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WxFactory"
copyright = "2024, ECCC"
author = "ECCC"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_mdinclude",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autosummary_mock_imports = ["wx_cupy", "precondition.preconditioner_dg", "precondition.preconditioner_fv"]
autodoc_default_options = {
    # "members": True,
    # "undoc-members": True,
    # "private-members": True
}

templates_path = ["_templates"]
exclude_patterns = []


def skip(app, what, name, obj, would_skip, options):
    if name in ["__call__", "__init__", "__compute_rhs__"]:
        return False
    elif name[:2] == "__":
        return True
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]
