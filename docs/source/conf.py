# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

root = Path(__file__).parents[2]
print("root", root)
sys.path.insert(1, str(root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyOlimp"
copyright = "2025, PyOlimp authors"
author = "PyOlimp authors"
from olimp import __version__

release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx-jsonschema",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# see https://alabaster.readthedocs.io/en/latest/customization.html#theme-options
html_theme_options = {
    "sidebar_width": "17em",
    "page_width": "90%",
}

root = Path(__file__).parents[1]
from docs.gen_images import gen_images

print("generating images")
gen_images()
print("done")
