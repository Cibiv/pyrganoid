# rham

Hamming distance for python polars DataFrames written in rust for performance.

## Installation

Install maturin and compile via `maturin build --release`, which creates a python wheel in `dist/`.

## Usage

Usage example

    import polars as pl
    import rham

    df1 = pl.DataFrame({"lid": ["AACG", "ACGG"]})
    df2 = pl.DataFrame({"lid": ["ACCC", "GGGG"]})
    col1 = "lid"
    col2 = "lid"

    max_hamming_distance = max(df1.select(pl.col(col1).str.lengths().max())[col1][0], df2.select(pl.col(col2).str.lengths().max())[col2][0])
    rham.compute_hamming(df1, df2, col1, col2, max_hamming_distance)
