Creates hamming visualisations (figure 2d)

# Installation

First compile rham (see `rham/README.md` for instructions). Then install the compiled rham package, pyarrow, polars, bokeh and selenium.

# Running

Download and decompress the file `GSM6599035_lt48.csv` and place it into `../data/`. The file is contained in the file `GSE214105_RAW.tar` from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214105 .

First run `python generate_data.py` which generates the file `hamming_distances.csv`. Then run `python vis.py` which creates the hamming distance plots for each replicate and library.
