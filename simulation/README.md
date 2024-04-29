# Installation

Install `pyrganoid`, `remoter`, `syn_bokeh_helpers` (which can be find in `./dependencies`) and all dependencies listed in `requirements.txt`.

# Running

First run all calculations, preferrably on a cluster. You can start the orchestrator via `python -m simulations.cli`. Execute `python -m remoter.cli worker` to start a worker.

Generate figures via `python -m simulations.cli --plot`, which takes the data from `.remoter.sqlite3` and the `remoter` directory. If you ran the computations on a different computer, copy these over first.

If you encounter the error `TypeError: did not expect type: 'bool' in 'concat'` this means that a computation was not precomputed. Restart the program with the same parameters and the error should not appear again.

