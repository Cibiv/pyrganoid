Creates hamming visualisations (figure 2d)

# Installation

First compile rham (see `rham/README.md` for instructions). Then install the compiled rham package, pyarrow, polars, bokeh and selenium.

# Running

Ensure `LT47.parquet` and `LT48.parquet` are present in `../data` (check README there for instructions).
Run `python generate_data.py` which generates the file `hamming_distances.csv`. Then run `python vis.py` which creates the hamming distance plots for each replicate and library.
