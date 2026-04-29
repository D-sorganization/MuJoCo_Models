# SPDX-License-Identifier: MIT
# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "MuJoCo Models"
copyright = "2026, D-sorganization"
author = "D-sorganization"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Autodoc options ---------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
