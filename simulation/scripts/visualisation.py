import numpy as np
from os.path import join as pjoin


from bokeh.models import Plot
from bokeh.io import show, export_svg
from bokeh.layouts import grid
from bokeh.plotting import figure
from bokeh.palettes import plasma

from theme import theme

def syn_save(p, output=None, outputtype="svg", save=True, jsontag=None, header="", body=""):
    """Save a bokeh plot as svg, json or html"""
    if outputtype not in ["svg", "json", "html"]:
        raise ValueError(f"outputtype is {outputtype}, but must be one of svg, json or html.")
    if jsontag and (outputtype != "json"):
        raise ValueError(f"Jsontag provided, but outputtype is {outputtype}, not json")
    if (not output) and save :
        raise ValueError("No output provided, but saving requesed")
    if not save:
        raise Exception("Not supported in this version of syn_save")
    elif outputtype == "html":
        raise Exception("Not supported in this version of syn_save")
    elif outputtype == "svg":
        _make_backend_svg(p)
        if not output.endswith(".svg"):
            output += ".svg"
        export_svg(p, filename=output)
    elif outputtype == "json":
        raise Exception("Not supported in this version of syn_save")


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


def pixel_filter(reads, ranks, glyph_size, frame_width, frame_height, x_range, y_range, pixel_overlap):
    """Filter datapoints from a ranked log-log plot that are closer than 1 pixel"""
    lranks, lreads = np.log10(ranks), np.log10(reads)
    x_pixel = frame_width * (lranks - np.log10(x_range[0])) / (np.log10(x_range[1]) - np.log10(x_range[0]))
    y_pixel = frame_height * (lreads - np.log10(y_range[0])) / (np.log10(y_range[1]) - np.log10(y_range[0]))

    mask = np.ones(len(x_pixel), dtype=bool)

    xo, yo = np.inf, np.inf

    for idx, (x, y) in enumerate(zip(x_pixel, y_pixel)):
        dist = np.sqrt((x-xo)**2 + (y-yo)**2)
        if dist > pixel_overlap * glyph_size:
            xo, yo = x, y
        else:
            mask[idx] = False
    mask[0] = True
    mask[-1] = True
    return reads[mask], ranks[mask]



def plot_ranked(reads, color="blue", p=None, glyph_size=1, frame_width=300, frame_height=200,x_range=None, y_range=None, pixel_overlap=0.25, **kwargs):
    if x_range is None:
        x_range = (1, len(reads))
    if y_range is None:
        y_range = (np.min(reads), np.max(reads))

    reads = np.sort(reads)[::-1]
    ranks = np.arange(len(reads)) +1

    if p is None:
        p = figure(frame_width=frame_width, frame_height=frame_height, x_range=x_range, y_range=y_range, **kwargs)

    reads, ranks = pixel_filter(reads, ranks, glyph_size, frame_width, frame_height, x_range, y_range, pixel_overlap)

    p.circle(ranks, reads, size=glyph_size, color=color)

    return p


def plot_all(dats, root, title, font_size=5):
    colorscale = plasma(len(dats)+1)[:-1][::-1]

    p = None
    for color, day in zip(colorscale, dats):
        p = plot_ranked(dats[day], x_axis_label='Rank', y_axis_label='Lineagesize', frame_width=200, frame_height=200, color=color, p=p, glyph_size=1, x_axis_type='log', y_axis_type='log', y_range=(4, 2*10**6), x_range=(0.5, 2*10**4))
    theme(p, font_size=font_size)
    syn_save(p, pjoin(root, f'{title}_ranked.svg'))


def plot_celltype_composition(out, root, title, font_size=5):

    colors = {'S': '#6495ed', 'A': '#ffb90f', 'N': '#6e8b3d'}

    all = sum(out[celltype] for celltype in out['celltypes'])

    p = figure(frame_width=200, frame_height=200, x_range=(0,40), y_range=(-0.01, 1.01))
    for celltype in out['celltypes']:
        p.line(x=out['time'], y=out[celltype]/all, color=colors[celltype.name])
    theme(p, font_size=font_size)

    syn_save(p, pjoin(root, f'{title}_celltype_composition.svg'))


def plot_celltypes_ranked(day, dats, out, root, title, font_size=5, plot_n=True):

    i = np.searchsorted(out['time'], day)
    if i == len(out["time"]):
        i = i-1
    colors = {'S': '#6495ed', 'A': '#ffb90f', 'N': '#6e8b3d'}

    celltypes = [out['lineages'][:, i, j] for j in range(len(out['celltypes']))]

    cells = np.sum(out['lineages'][:, i, :], axis=1)
    factor = 10 / np.min(cells)
    print('cells.shape', cells.shape)

    p = None

    for d, c in zip(celltypes, out['celltypes']):
        if c.name == 'N' and not plot_n:
            continue
        d = np.sort(d)[::-1]
        p = plot_ranked(factor*d, x_axis_label='Rank', y_axis_label='Lineagesize', frame_width=200, frame_height=200, color=colors[c.name], p=p, glyph_size=1, x_axis_type='log', y_axis_type='log', y_range=(4, 2*10**6), x_range=(0.5, 2*10**4))

    p = plot_ranked(10*dats[day] / np.min(dats[day]), x_axis_label='Rank', y_axis_label='Lineagesize', frame_width=200, frame_height=200, color='black', p=p, glyph_size=1, x_axis_type='log', y_axis_type='log', y_range=(4, 2*10**6), x_range=(0.5, 2*10**4))


    theme(p, font_size=font_size)
    syn_save(p, pjoin(root, f'{title}_ranked_celltypes_day{day}.svg'))

