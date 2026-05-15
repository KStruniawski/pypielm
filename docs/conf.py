# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Suppress PendingDeprecationWarnings from sphinx-autodoc-typehints on Sphinx 9
warnings.filterwarnings(
    "ignore",
    message=".*set_application.*is deprecated",
    category=DeprecationWarning,
)

# Make the package importable without installing
sys.path.insert(0, str(Path(__file__).parents[1]))

import pypielm  # noqa: E402

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "PyPIELM"
copyright = "2026, Karol Struniawski"
author = "Karol Struniawski"
release = pypielm.__version__
version = pypielm.__version__

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # Google-style docstrings
    "sphinx.ext.viewcode",          # [source] links
    "sphinx.ext.intersphinx",       # cross-links to Python / NumPy / Torch
    "sphinx.ext.mathjax",           # LaTeX math in docstrings
    "sphinx_autodoc_typehints",     # type annotations in signatures
    "myst_parser",                  # Markdown support
]

# autosummary: generate stub .rst files automatically
autosummary_generate = True

# napoleon settings (Google-style)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "both"

# Suppress known false-positive warnings
suppress_warnings = [
    "sphinx_autodoc_typehints.local_function",   # closures without @functools.wraps
    "sphinx_autodoc_typehints.forward_reference", # private TypeVars not resolvable
    "ref.duplicate",                               # autosummary re-documents symbols
]

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# MyST markdown extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",          # $...$ inline math
    "amsmath",             # $$...$$ display math
    "smartquotes",
]
myst_heading_anchors = 3

# Source suffixes: both .rst and .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = f"PyPIELM {release}"
html_static_path = ["_static"]
html_css_files = []

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/kstruniawski/pypielm",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/kstruniawski/pypielm",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# ---------------------------------------------------------------------------
# LaTeX / PDF output
# ---------------------------------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
}
latex_documents = [
    ("index", "pypielm.tex", "PyPIELM Documentation", author, "manual"),
]
