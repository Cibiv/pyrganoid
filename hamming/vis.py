from bokeh.io import show
from bokeh.plotting import figure
import polars as pl

import numpy as np

from bokeh.models import Plot
from bokeh.io import export_svg

import os.path


def _make_backend_svg(p):
    if isinstance(p, Plot):
        p.output_backend = "svg"
    else:
        if hasattr(p, "children"):
            for child in p.children:
                _make_backend_svg(child)
        else:
            try:
                for child in p:
                    _make_backend_svg(child)
            except:
                pass


def save(p, output=None):
    """Save a bokeh plot as svg"""
    _make_backend_svg(p)
    if not output.endswith(".svg"):
        output += ".svg"
    export_svg(p, filename=output)


def style(p):
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.outline_line_color = None
    p.border_fill_color = None
    p.background_fill_color = None
    p.xaxis.axis_label_text_font = "helvetica"
    p.yaxis.axis_label_text_font = "helvetica"
    p.xaxis.axis_label_text_color = "black"
    p.yaxis.axis_label_text_color = "black"
    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_style = "normal"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.axis_label_text_font_size = "10pt"
    p.xaxis.major_label_text_font_size = "8pt"
    p.yaxis.major_label_text_font_size = "8pt"
    p.xaxis.major_label_text_color = "black"
    p.yaxis.major_label_text_color = "black"
    p.title.text_font_size = "10pt"
    p.title.text_font = "helvetica"
    p.title.text_color = "black"
    p.title.text_font_style = "bold"


def plot_circles_linear(data, p=None):
    if p is None:
        p = figure(y_range=(0, 0.30), x_range=(0, 50), frame_width=200, frame_height=200, x_axis_label='Hamming Distance', y_axis_label='Percentage')
    p.scatter(x='distance', y='count', source=data.to_pandas(), size=2, color="black")
    p.line(x='distance', y='count', source=data.to_pandas(), color="black")
    style(p)
    return p


df = pl.read_csv('./hamming_distances.csv')

rs = df.select("r1", "r2").unique()
d = df.select("r1", "r2", "l1", "l2").unique()

os.makedirs("figures", exist_ok=True)

for idx ,(r1, r2, l1, l2) in enumerate(d.iter_rows()):
    data = df.filter( (pl.col("r1") == r1) & (pl.col("r2") == r2) & (pl.col("l1") == l1) & (pl.col("l2") == l2))
    data = data.filter(pl.col("distance") != 0)
    data = data.with_columns(pl.col('count') / pl.col('count').sum())

    p = plot_circles_linear(data)

    save(p, f"figures/hamming_distances_r1={r1}_r2={r2}_l1={l1}_l2={l2}.svg")

