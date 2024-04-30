import functools
import os
import os.path
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import tee

import bokeh
import bokeh.resources
import matplotlib.pylab as pl
import numpy as np
import ot
import ot.plot
import pandas as pd
import pyarrow
import pyarrow as pa
import pyarrow.compute
import pyarrow.csv
import pyarrow.parquet
import scipy
import scipy.stats
from bokeh.core.properties import field, value
from bokeh.io import export_svgs, output_notebook, show
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import ColorBar, ColumnDataSource, CustomJSTransform, Whisker
from bokeh.palettes import viridis
from bokeh.plotting import figure
from bokeh.plotting import save as bokeh_save
from bokeh.transform import dodge, linear_cmap, stack
from ot.datasets import make_1D_gauss as gauss

from syn_bokeh_helpers import syn_save

LINE_WIDTH = 0.66
SIZE = 3

NegateTransform = CustomJSTransform(
    func="return -x",
    v_func="""
var new_xs = new Array(xs.length)
for(var i = 0; i < xs.length; i++) {
    new_xs[i] = -xs[i]
}
return new_xs
""",
)


def negate(field_name):
    return field(field_name, NegateTransform)


def _set_default(figure_kwargs, **kwargs):
    """Set default values for dicts"""
    for key, value in kwargs.items():
        if key not in figure_kwargs:
            figure_kwargs[key] = value


def _make_source(*sources):
    r = {}
    for source in sources:
        if source:
            r.update(source)
    return ColumnDataSource(r)


def save(obj, filename, title="Organoid Model"):
    prefix = os.getenv("SYN_OUTPUT_DIR", "")
    if os.getenv("SYN_SVG", False):
        backend = obj.output_backend
        obj.output_backend = "svg"
        export_svgs(obj, prefix + filename + ".svg")
        obj.output_backend = backend
    else:
        bokeh_save(
            obj,
            os.path.join(prefix, filename + ".html"),
            resources=bokeh.resources.INLINE,
            title=title,
        )


def celltypes_over_time(
    result, p=None, colors={}, legend_location="bottom_right", **kwargs
):
    """Plot number of cells for each celltype and their total"""
    _set_default(
        kwargs, y_axis_type="log", y_range=(1, 10_000_000), width=300, height=300
    )
    if p is None:
        p = figure(**kwargs)

    data = {}
    data["time"] = result["time"]
    for celltype in result["celltypes"]:
        data[celltype.name] = result[celltype]

    source = ColumnDataSource(data)

    celltype_names = [celltype.name for celltype in result["celltypes"]]

    for name in celltype_names:
        p.line(
            x="time",
            y=name,
            legend_label=name,
            color=colors.get(name, "blue"),
            source=source,
        )

    p.line(
        x="time",
        y=stack(*celltype_names),
        legend_label="Total",
        color="black",
        source=source,
    )

    p.legend.location = legend_location

    return p


def line(p, time, measurements, **kwargs):
    """Plot a averaged line with standard deviation for multiple measurements

    p : bokeh.Plot
    time : np.array (1 dim)
    measurements : np.array (2 dim) - First dimension time, second dimension measurements
    **kwargs : dict - passed on to p.line
    """
    x = time
    ys = measurements
    average = np.average(ys, axis=0)
    p.line(x, average, **kwargs)
    for x_coordinate in x:
        p.line

    standard_deviation = np.std(ys, axis=0)
    whisker_source = {
        "base": x,
        "upper": average + standard_deviation,
        "lower": average - standard_deviation,
    }
    w = Whisker(
        source=ColumnDataSource(whisker_source),
        base="base",
        upper="upper",
        lower="lower",
        **kwargs,
    )

    p.add_layout(w)

def violin_scaled_all(xs, ys, p=None, color1="", color2="", width_dodge=0.4, **kwargs):
    ns = []
    for y1, y2 in ys:
        if len(y1) == 0 and len(y2) == 0:
            continue
        y1f, y2f = kde2(y1, y2)[2:4]
        m = max(np.max(y1f) if len(y1f) > 0 else -np.inf, np.max(y2f) if len(y2f) > 0 else -np.inf)
        ns.append(m)
    max_all = max(ns)

    #max_all = max(max(np.max(yf1), np.max(yf2)) for y1, y2 in ys for yf1, yf2 in [kde2(y1, y2)[2:4]])

    for x, (y1, y2) in zip(xs, ys):
        if len(y1) == 0 and len(y2) == 0:
            continue
        _, _, freq1, freq2 = kde2(y1, y2)
        m1, m2 = np.max(freq1), np.max(freq2)
        m = max(m1, m2)
        wd = width_dodge * (m / max_all)
        p = violin2(x, y1, y2, p=p, color1=color1, color2=color2, width_dodge=wd, **kwargs)
    return p

def violin2(x, y1, y2, p=None, color1="gray", color2="blue", width_dodge=0.4, scaled=True, **kwargs):
    _, _, freq1, freq2 = kde2(y1, y2)
    m1, m2 = np.max(freq1), np.max(freq2)
    m = max(m1, m2)
    if scaled:
        w1, w2 = 2* width_dodge * (m1/m), 2*width_dodge * (m2/m)
    else:
        w1, w2 = width_dodge, width_dodge
    p = violin(x, y1, p=p, fillcolor=color1, width_dodge=w1, kind="left", **kwargs)
    p = violin(x, y2, p=p, fillcolor=color2, width_dodge=w2, kind="right", **kwargs)
    return p

def violin_calc_data(y, num):
    mean = np.mean(y)
    median = np.median(y)

    y = y[y != 0]
    log_y = np.log(y)
    positions = np.exp(np.linspace(np.min(log_y), np.max(log_y), num=num))
    positions = np.concatenate([np.array([np.min(positions)]), positions, np.array([np.max(positions)])])
    y_kde = scipy.stats.gaussian_kde(log_y)
    frequency = y_kde(np.log(positions))


def violin(x, y, y_axis_type="log", width_dodge=0.4, p=None, kind="both", fillcolor="lightgray", num=100, legend_label="", lib1=None, lib2=None, **kwargs):
    """
    y_axis_type : Either("linear", "log", "datetime")
    """

    kwargs["y_axis_type"] = y_axis_type
    _set_default(kwargs, width=300, height=300)

    if p is None:
        p = figure(**kwargs)

    # mean should be calculated before applying the log
    mean = np.mean(y)
    median = np.median(y)

    if y_axis_type == "log":
        y = y[y != 0]
        log_y = np.log(y)
        positions = np.exp(np.linspace(np.min(log_y), np.max(log_y), num=num))
        positions = np.concatenate([np.array([np.min(positions)]), positions, np.array([np.max(positions)])])
        y_kde = scipy.stats.gaussian_kde(log_y)

        frequency = y_kde(np.log(positions))
    else:
        positions = np.exp(np.linspace(np.min(y), np.max(y), num=num))
        y_kde = scipy.stats.gaussian_kde(y)

        frequency = y_kde(positions)

    frequency[0] = 0
    factor = np.max(frequency) / width_dodge
    frequency = frequency / factor 
    frequency[-1] = 0

    x_range_value = x
    if not isinstance(x_range_value, Iterable) or isinstance(x_range_value, str):
        x_range_value = [x_range_value]

    x_right = [(*x_range_value, f if kind in ["right", "both"] else 0) for f in frequency]
    x_left = [(*x_range_value, -f if kind in ["left", "both"] else 0) for f in frequency]

    p.harea(
        x1=x_right, x2=x_left, y=positions, fill_color=fillcolor
    )
    p.line(x=x_right, y=positions, line_color="black", line_width=LINE_WIDTH)
    p.line(x=x_left, y=positions, line_color="black", line_width=LINE_WIDTH)
    p.line(
        x=[x_right[0], x_left[0]], y=[positions[0], positions[0]], line_color="black", line_width=LINE_WIDTH
    )

    t = np.log if y_axis_type == "log" else (lambda x: x)

    lower_percentile = np.percentile(y, 25)
    low_left = float(y_kde(t(lower_percentile)) / factor)
    lp_x = [(*x_range_value, -low_left if kind in ["left", "both"] else 0) , (*x_range_value, low_left if kind in ["right", "both"] else 0 )]

    higher_percentile = np.percentile(y, 75)
    high_left = float(y_kde(t(higher_percentile)) / factor)
    hp_x = [(*x_range_value, -high_left if kind in ["left", "both"] else 0) , (*x_range_value, high_left if kind in ["right", "both"] else 0 )]

    p.line(x=lp_x, y=[lower_percentile, lower_percentile], color="black", line_width=0.5)
    p.line(x=hp_x, y=[higher_percentile, higher_percentile], color="black", line_width=0.5)

    if kind == "right":
        dd = width_dodge/2
    elif kind == "left":
        dd = -width_dodge/2
    else:
        dd = 0

    p.circle(x=[(*x_range_value, dd)], y=median, color="black", size=SIZE, legend_label="median", line_width=LINE_WIDTH)
    p.circle(
        x=[(*x_range_value, dd)],
        y=mean,
        fill_color="white",
        line_color="black",
        size=SIZE,
        legend_label="mean",
        line_width=LINE_WIDTH
    )

    p.legend.location = "top_left"

    return p


def gini_curve(
    data: np.ndarray,
    line_color="black",
    p: bokeh.models.Plot = None,
    legend_label=None,
    source=None,
    **kwargs,
):
    """Plot the normalized sorted cummulative sum of data in p with color"""

    _set_default(
        kwargs,
        width=300,
        height=300,
        x_axis_label="Fraction of Lineages",
        y_axis_label="Fraction of Cells",
    )

    if p is None:
        p = figure(**kwargs)

    yy = data
    yy = np.sort(yy)
    yy = np.cumsum(yy)
    yy = yy / np.max(yy)
    xx = np.linspace(0, 1, len(yy))

    def filter_by_distance(state, point, epsilon=0.01):
        "Extend state by point if point has more than epsilon distance from the last state entry"
        x1, y1 = state[-1]
        x2, y2 = point
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 >= epsilon:
            state.append((x2, y2))
        return state

    x, y = list(
        zip(
            *functools.reduce(filter_by_distance, zip(xx, yy), [(xx[0], yy[0])]),
            (xx[-1], yy[-1]),
        )
    )

    source = _make_source(source, {"x": x, "y": y})
    p.line(
        x="x",
        y="y",
        line_color=line_color,
        legend_label=legend_label,
        source=source,
        muted_alpha=0.2,
    )

    p.legend.click_policy = "hide"
    legend = p.legend[0]
    legend.label_text_font_size = "6pt"
    legend.padding = 1
    legend.spacing = 0
    legend.glyph_height = 10
    legend.label_height = 0
    legend.location = "top_left"
    legend.visible = False
    p.xaxis[0].axis_label_standoff = 0
    p.yaxis[0].axis_label_standoff = 0
    p.xaxis[0].axis_label_text_font_size = "8pt"
    p.yaxis[0].axis_label_text_font_size = "8pt"
    p.xaxis[0].major_label_text_font_size = "4pt"
    p.yaxis[0].major_label_text_font_size = "6pt"
    return p


def percentiles(
    time, data, percentiles=20, p=None, line_color="black", source=None, **kwargs
):
    _set_default(
        kwargs, x_range=(min(time), max(time)), y_range=(0, 1), width=300, height=300
    )
    if p is None:
        p = figure(**kwargs)

    data = np.sort(data, axis=0)
    cum_data = np.cumsum(data[::-1, :], axis=0)
    cum_data = cum_data / cum_data[-1, :][None, :]

    linlen, timelen = cum_data.shape

    for i in np.linspace(0, linlen - 1, percentiles):
        day_source = _make_source(source, {"time": time, "data": cum_data[int(i), :]})
        p.line(x="time", y="data", line_color=line_color, source=day_source)

    p.xaxis[0].axis_label_standoff = 0
    p.yaxis[0].axis_label_standoff = 0
    p.xaxis[0].axis_label_text_font_size = "8pt"
    p.yaxis[0].axis_label_text_font_size = "8pt"
    p.xaxis[0].major_label_text_font_size = "6pt"
    p.yaxis[0].major_label_text_font_size = "6pt"

    return p


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def kde2(d1, d2, x=None, num=200):
    m = min(np.min(d1) if len(d1) > 0 else np.inf, np.min(d2) if len(d2) > 0 else np.inf)
    mmax = max(np.max(d1) if len(d1) > 0 else -np.inf, np.max(d2) if len(d2) > 0 else -np.inf)

    if x is None:
        x = np.linspace(start=m, stop=mmax, num=num)

    k1 = scipy.stats.gaussian_kde(d1)(x) if len(d1) > 0 else []
    k2 = scipy.stats.gaussian_kde(d2)(x) if len(d2) > 0 else []

    return (m, mmax, k1, k2)


def kde(data, x=None, num=200):
    m = np.min(data)
    mmax = np.max(data)
    
    k = scipy.stats.gaussian_kde(data)
    
    if x is None:
        x = np.linspace(start=m, stop=mmax, num=num)

    kx = k(x)
    kx[x<m] = 0
    kx[x>mmax] = 0
    
    return (x, kx)

def emd(s1, s2, counts=300) -> bokeh.plotting.figure:
    d1 = np.log(s1)
    d2 = np.log(s2)

    x_range = np.linspace(min(np.min(d1), np.min(d2)), max(np.max(d1), np.max(d2)), counts)

    x1, y1 = kde(d1, x_range)
    x2, y2 = kde(d2, x_range)

    y1, y2 = y1 / np.sum(y1), y2 / np.sum(y2)

    M = ot.dist(x1.reshape((len(x1), 1)), x1.reshape((len(x1), 1)))
    Gs = ot.emd(y1, y2, M)
    
    cost = np.sum(M * Gs)
    
    return np.exp(x_range), y1, y2, M, Gs, cost

def visualize_emd(x, y1, y2, M, Gs, cost, width=200, height=200, dist_height=70, names=[None, None], bounds=True) -> bokeh.plotting.figure:
    
    p_heatmap = figure(frame_width=width, frame_height=height, name="heatmap",
                       x_range=(x[0], x[-1]), y_range=(x[0], x[-1]), x_axis_type="log", y_axis_type="log")
    
    if bounds:
        p_heatmap.x_range.bounds = x[0], x[-1] 
        p_heatmap.y_range.bounds = x[0], x[-1] 

        
    display = Gs.copy()
    #display = np.flip(display, axis=0)
    #display = np.flip(display, axis=1)

    display[display == 0] = float("nan")
    
    cmap = linear_cmap("", low=np.min(Gs), high=np.max(Gs), palette=viridis(256), nan_color="#00000000")
    
    cds = ColumnDataSource({"image":[display]})
    p_heatmap.image(image="image", x=x[0], y=x[0], dw=x[-1]-x[0], dh=(x[-1]-x[0]),
                    color_mapper=cmap["transform"], source=cds, name="image")
    
    p1 = figure(frame_height=height, frame_width=dist_height, y_range=p_heatmap.y_range, 
                y_axis_label=names[0],  y_axis_type="log")
    p1.line(y1, x)

    p2 = figure(frame_height=dist_height, frame_width=width, x_range=p_heatmap.x_range, 
                x_axis_location="below", x_axis_label=names[1],x_axis_type="log")
    p2.line(x, y2)
    
    
    
    p1.xaxis.visible = False
    p2.yaxis.visible = False
    p_heatmap.xaxis.visible = False
    p_heatmap.yaxis.visible = False
    
    color_bar = ColorBar(color_mapper=cmap["transform"],
                     label_standoff=12, border_line_color=None, location=(0,0))
    
    p_heatmap.add_layout(color_bar, 'right')
    
    p_heatmap.line(x=[x[0], x[-1]], y=[x[0], x[-1]], color="red")


    weights = Gs / np.sum(Gs, axis=0).reshape(1, Gs.shape[0])
    T = (x @ weights)

    p_heatmap.line(x, T, color="black")


    return gridplot([[p1, p_heatmap], [None, p2],], toolbar_location="right")
    
    
def original_vis(s1, s2):
    x, a, b, M, G0, cost = emd(s1, s2)
    pl.figure(3, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')

def dists(datasets, labels, titles, widths=(150,200, 200), heights=200, ranked_colors=None, legends=True) -> bokeh.plotting.figure:
    _c = 3
    if not isinstance(titles, Iterable):
        titles = [titles]*_c
    if not isinstance(widths, Iterable):
        widths = [widths]*_c
    if not isinstance(heights, Iterable):
        heights = [heights]*_c
    if not isinstance(legends, Iterable):
        legends = [legends]*_c
        
    if ranked_colors is None:
        ranked_colors = viridis(len(labels))
        
    p_violins = violins(datasets, labels, title=titles[0], width=widths[0], height=heights[0])
    p_ranked = ranked(datasets, labels, title=titles[1], width=widths[1], height=heights[1], colors=ranked_colors)

    p_violins.legend.visible = legends[0]
    p_ranked.legend.visible = legends[1]
    e = emd(*datasets, 200)
    p_emd = visualize_emd(*e, names=labels, width=widths[2], height=heights[2], bounds=False) # type: ignore

    return row(children=[p_violins, p_ranked, p_emd])

def ranked(datasets, labels, colors, title, width, height=300) -> bokeh.plotting.figure:
    
    ds = []
    mm = 0
    for d in datasets:
        if isinstance(d, np.ndarray) or isinstance(d, pd.Series):
            dd = d
        else:
            dd = d["nreads"]
        ds.append(dd)
        mm = max(mm, np.max(dd))
        
    p = figure(title=title, frame_width=width, x_axis_type="log", x_axis_label="Rank", y_axis_label="Reads",
               y_axis_type="log", x_range=(1, mm), frame_height=height)

        
    for d, l, c in zip(ds, labels, colors):
        y = sorted(d)[::-1]
        x = np.arange(1, len(y)+1)
        p.line(x=x, y=y, color=c, legend_label=l)
    return p

def violins(datasets, labels, title, width, height) -> bokeh.plotting.figure:
    p = None
    for d, l in zip(datasets, labels):
        if isinstance(d, np.ndarray) or isinstance(d, pd.Series):
            dd = d
        else:
            dd = d["nreads"]
        p = violin(l, dd, x_range=labels, frame_width=width, frame_height=height, title=title, p=p)
    return p

        
        
