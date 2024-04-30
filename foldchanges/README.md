# Foldchange analysis

The algorithm to calculate the foldchanges can be found in synot.py:no_kde_algorithm

# Installation

Install all dependencies listed in `requirements.txt` as well as `remoter` and `syn_bokeh_helpers` (both found in `../simulation/dependencies/`).

Then compile the coconut file by executing `coconut main.coco`, which generates the file `main.py`.

# Running

Run the following command, which starts the orchestrator:

    python main.py --threshold=0.99 --show-dens --frame-width=300 --frame-height=150 svg/all.svg

Also start at least one worker via

    python -m remoter.cli worker
