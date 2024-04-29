from bokeh.io import show as bokeh_show, save as bokeh_save, export_svg
from bokeh.models import Plot, CustomJSTransform, ColumnDataSource, CustomJS
from bokeh.embed import json_item
from bokeh.models.layouts import GridBox
from bokeh.transform import transform, factor_cmap, dodge
from bokeh.core.properties import field, value
import bokeh.resources
import json
import math
import collections
from itertools import tee
import importlib.resources as pkg_resources
import jinja2
import numpy as np

TEMPLATE = pkg_resources.read_text(__package__, "template.jinja2")


def general_mapper(column, m):
    v_func = """
    const first = xs[0]
    const norm = new Float64Array(xs.length)
    for (let i = 0; i < xs.length; i++) {
        norm[i] = m[xs[i]];
    }
    return norm
    """
    t = CustomJSTransform(args={"m": m}, v_func=v_func)

    return transform(column, t)


def color_mapper(column, m):
    return factor_cmap(column, list(m.values()), list(m.keys()))


def rows_to_columns(tuples, names):
    r = {name: [] for name in names}
    for t in tuples:
        for name, value in zip(names, t):
            r[name].append(value)
    return r


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

def embed_callback(j, tag, source):
    cb = CustomJS(args={"source": source}, code=f"""
            const idx = Math.min(source.selected.indices);
            const j = source.data['{j}'][idx];
            const tag = source.data['{tag}'][idx];
            syn_embed(j, tag);
            """)
    return cb


def negate(field_name):
    return field(field_name, NegateTransform)


def set_default(figure_kwargs, **kwargs):
    """Set default values for dicts"""
    for key, value in kwargs.items():
        if key not in figure_kwargs:
            figure_kwargs[key] = value


def unify_sources(*sources):
    r = {}
    for source in sources:
        if source:
            r.update(source)
    return ColumnDataSource(r)


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


def syn_save(p, output=None, outputtype="svg", save=True, jsontag=None, header="", body=""):
    """Save a bokeh plot as svg, json or html"""
    if outputtype not in ["svg", "json", "html"]:
        raise ValueError(f"outputtype is {outputtype}, but must be one of svg, json or html.")
    if jsontag and (outputtype != "json"):
        raise ValueError(f"Jsontag provided, but outputtype is {outputtype}, not json")
    if (not output) and save :
        raise ValueError("No output provided, but saving requesed")

    if not save:
        bokeh_show(p)
    elif outputtype == "html":

        template = TEMPLATE
        template = template.replace("SYN_REPLACEME_BODY", body)
        template = template.replace("SYN_REPLACEME_HEADER", header)
        template_j= jinja2.Environment(loader=jinja2.BaseLoader()).from_string(template)
        bokeh_save(p, filename=output, template=template_j, resources=bokeh.resources.INLINE)
    elif outputtype == "svg":
        _make_backend_svg(p)
        if not output.endswith(".svg"):
            output += ".svg"
        export_svg(p, filename=output)
    elif outputtype == "json":
        with open(output, mode="w") as f:
            json.dump(json_item(p, jsontag), f)


def fix_axis_in_grid(grid):

    row_ranges = collections.defaultdict(list)
    column_ranges = collections.defaultdict(list)

    most_left = collections.defaultdict(lambda: (None, math.inf, math.inf))
    most_down = collections.defaultdict(lambda: (None, -1, -1))

    for p, row, column in grid.children:
        candidate_p, candidate_row, candidate_column = most_left[row]
        if candidate_column > column:
            most_left[row] = (p, row, column)

        candidate_p, candidate_row, candidate_column = most_down[column]
        if candidate_row < row:
            most_down[column] = (p, row, column)

        p.xaxis.visible = False
        p.yaxis.visible = False

        row_ranges[row].append(p)
        column_ranges[column].append(p)


    for p, _, _ in most_left.values():
        p.yaxis.visible = True

    for p, _, _ in most_down.values():
        p.xaxis.visible = True

    for v in row_ranges.values():
        r = None
        for p in v:
            if r is None:
                r = p.y_range
            else:
                p.y_range = r

    for v in column_ranges.values():
        r = None
        for p in v:
            if r is None:
                r = p.x_range
            else:
                p.x_range = r


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def pixel_filter(x, y, nx, ny, glyph_size, frame_width, frame_height, x_range, y_range, pixel_overlap):
    x_pixel = frame_width * (nx - x_range[0]) / (x_range[1] - x_range[0])
    y_pixel = frame_height * (ny - y_range[0]) / (y_range[1] - y_range[0])

    mask = np.ones(len(x_pixel), dtype=bool)

    xo, yo = np.inf, np.inf

    for idx, (xx, yy) in enumerate(zip(x_pixel, y_pixel)):
        dist = np.sqrt((xx-xo)**2 + (yy-yo)**2)
        if dist > pixel_overlap * glyph_size:
            xo, yo = xx, yy
        else:
            mask[idx] = False
    mask[0] = True
    mask[-1] = True
    return x[mask], y[mask]


