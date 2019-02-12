# FACS Analysis

The Jupyter notebook in this repo, `facs_analysis.ipynb`, walks you through the basic steps of analyzing Mass-Titr FACS results. For high-throughput binding experiments, the `highThroughputScripts` pipeline is available on [GitHub](https://github.com/KeatingLab/highThroughputScripts/tree/vs_optimization).

To complete the analysis, run all cells in order one-by-one unless otherwise specified, filling in or changing input values as needed. To export a plot, you may want to change the DPI in the `plt.figure` command, i.e. `plt.figure(..., dpi=160)`.

**Requirements**: This notebook requires Python 2, and the following modules:

* `FlowCytometryTools` (install using `pip install flowcytometrytools`)
* `lmfit`
* `ipywidgets`, version 7.2 or later (upgrade if necessary)
* The script `facs_utils.py` must be in the same directory as this notebook.

After installing these modules, be sure to restart the notebook kernel to make sure the modules are available.

*Written by*: Venkatesh Sivaraman, February 2019, adapted from scripts by Dustin Whitney and Theresa Hwang
