# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
# make package importable
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cellstate-pred'
copyright = '2025, Ariadna Villanueva Marijuan'
author = 'Ariadna Villanueva Marijuan'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # Google/NumPy docstring support
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_nb',                   # MyST notebook support (includes myst_parser)
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "dollarmath",
]
myst_heading_anchors = 3

# MyST-NB configuration
nb_execution_mode = "off"  # Don't execute notebooks during build
nb_execution_allow_errors = True

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

# PyData theme configuration
html_theme_options = {
    "logo": {
        "text": "cellstate-pred",
    },
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
}

# Sidebar navigation
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

