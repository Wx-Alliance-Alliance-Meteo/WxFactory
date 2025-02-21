# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import pathlib
import sys
import itertools

root = pathlib.Path(__file__).parents[3].resolve().as_posix()

sys.path.insert(0, root)
sys.path.append(os.path.join(root, "wx_factory"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WxFactory"
copyright = "2025, ECCC"
author = "ECCC"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
]

autosummary_generate = True
autosummary_mock_imports = [
    "precondition.preconditioner_dg",
    "precondition.preconditioner_fv",
    "rhs.rhs",
]
autodoc_default_options = {
    # "members": True,
    # "undoc-members": True,
    # "private-members": True
}
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []


def skip(app, what, name, obj, would_skip, options):
    if name in ["__call__", "__init__", "__compute_rhs__", "__enter__", "__exit__"]:
        return would_skip
    elif name[:2] == "__" or name[:1] == "_":
        return True
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def create_mandatory_directory():
    static = os.path.join(root, "doc", "sphinx", "source", "_static")
    os.makedirs(static, exist_ok=True)


create_mandatory_directory()
