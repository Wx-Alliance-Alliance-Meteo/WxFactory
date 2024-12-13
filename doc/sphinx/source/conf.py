# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import pathlib
import sys
import itertools

root = pathlib.Path(__file__).parents[3].resolve().as_posix()

sys.path.insert(0, root)

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
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
]

autosummary_generate = True
autosummary_mock_imports = ["wx_cupy", "precondition.preconditioner_dg", "precondition.preconditioner_fv", "rhs.rhs"]
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
    elif name[:2] == "__" or name[:1] == "_":
        return True
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]


def link_documentation():
    if not os.path.exists(root) or not os.path.isdir(root):
        raise ValueError("Something's wrong")

    def recursive_crawl(dirname: str) -> list[str]:
        if dirname == "doc/sphinx" or dirname == "tests/data" or dirname == ".git":
            return []

        actual_path = os.path.join(root, dirname)

        dirs = [dir for dir in os.listdir(actual_path) if os.path.isdir(os.path.join(actual_path, dir))]
        files = [
            os.path.join(dirname, file)
            for file in os.listdir(actual_path)
            if os.path.isfile(os.path.join(actual_path, file)) and os.path.splitext(file)[1] == ".md"
        ]
        files += list(
            itertools.chain.from_iterable([recursive_crawl(os.path.join(dirname, directory)) for directory in dirs])
        )

        return files

    content = [dir for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]
    documentation = list(itertools.chain.from_iterable([recursive_crawl(directory) for directory in content]))
    mirror = os.path.join(root, "doc", "sphinx", "source")

    for doc_file in documentation:
        mirror_doc = os.path.join(mirror, doc_file)
        true_doc = os.path.join(root, doc_file)
        if not (os.path.exists(mirror_doc) and os.path.islink(mirror_doc) and os.readlink(mirror_doc) == true_doc):
            if os.path.exists(mirror_doc):
                os.remove(mirror_doc)
            os.makedirs(os.path.dirname(mirror_doc), exist_ok=True)
            os.symlink(true_doc, mirror_doc)


link_documentation()
