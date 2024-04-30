import polars as pl

class LT48:

    def __init__(self, path="../data/LT48.parquet"):
        self.path = path
        self.data = None

    def load(self):
        self.data = pl.read_parquet(self.path)

    def get_data(self, prefix):
        df = self.data.filter((pl.col("category") == prefix))
        libs = sorted(df["effect"].unique())
        days = sorted(df["day"].unique())

        r = {}
        for day in days:
            r[day] = {}

            total = df.filter((pl.col("day") == day) )
            reps = sorted(total["replicate"].unique())
            for rep in reps:
                r[day][rep] = {lib: total.filter((pl.col("replicate") == rep) & (pl.col("effect") == lib))["nreads"].to_numpy() for lib in libs}
        return r

