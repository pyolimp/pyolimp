# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from typing import Literal, Any
from pathlib import Path
from sphinx.application import Sphinx

root = Path(__file__).parents[2]
print("root", root)
sys.path.insert(1, str(root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyOlimp"
copyright = "2025, PyOlimp authors"
author = "PyOlimp authors"
from olimp import __version__

module_version = release = __version__

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

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# see https://alabaster.readthedocs.io/en/latest/customization.html#theme-options
html_theme_options = {
    "sidebar_width": "17em",
}

html_css_files = [
    "custom.css",
]


def skip_forward_methods(
    app: Sphinx,
    what: Literal[
        "module", "class", "exception", "function", "method", "attribute"
    ],
    name: str,
    obj: Any,
    skip: bool,
    options: dict[str, bool],
) -> bool:
    # Skip default "forward" methods
    if name == "forward" and obj.__doc__ == None:
        return True
    return skip


def setup(app: Sphinx):
    app.connect("autodoc-skip-member", skip_forward_methods)


root = Path(__file__).parents[1]
from docs.gen_images import gen_images

print("generating images")
gen_images()
print("done")
