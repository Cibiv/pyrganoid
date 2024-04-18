import numpy as np
import polars as pl

from rham import compute_hamming

def main(data_path="../data/LT48.parquet", output_path="hamming_distances.csv", N=50_000):
    df = pl.scan_parquet(data_path)
    df = df.filter(pl.col('category') == 'LT-3D').filter(pl.col('day') == '00').collect()

    replicates = df["replicate"].unique().sort()

    dfs = []

    for r1 in replicates:
        for r2 in replicates:
            for l1, l2 in [ ("lib1", "lib1"), ("lib1", "lib2"), ("lib2", "lib1"), ("lib2", "lib2")]:
                d1 = df.filter(pl.col("replicate") == r1).filter(pl.col("effect") == l1)
                if len(d1) > N:
                    d1 = d1.sample(N)
                d2 = df.filter(pl.col("replicate") == r2).filter(pl.col("effect") == l2)
                if len(d2) > N:
                    d2 = d2.sample(N)
                x = compute_hamming(d1, d2, "lid", "lid", len(d1["lid"][0])+1)
                x = x.with_columns(pl.lit(r1).alias("r1"), pl.lit(r2).alias("r2"), pl.lit(l1).alias("l1"), pl.lit(l2).alias("l2"))
                dfs.append(x)

    result = pl.concat(dfs)

    result.write_csv(output_path)


if __name__ == "__main__":
    main()
