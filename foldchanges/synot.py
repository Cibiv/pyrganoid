from itertools import tee
from math import lcm

import numpy as np
import numpy.random
import scipy.stats
import ot
from bokeh.plotting import figure
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
from bokeh.colors import RGB

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.nonparametric.kernel_regression import KernelReg
import statsmodels.api as sm
from scipy.stats import norm

def gaussian(xs, x, scale=1):
    return np.exp(-(xs-x)**2/scale/2) / np.sqrt(2 * np.pi)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def weighted_kde2(x, dens, xs=None):
    kde = sm.nonparametric.KDEUnivariate(x)
    #kde.fit(kernel="gau", weights=dens, fft=False)
    kde.fit()
    if xs is None:
        xs = x
    return kde.evaluate(xs)

def weighted_kde(xx, dens, xs=None, wh=1000, scale=1):
    r = []
    for x in xs:
        i = np.searchsorted(xx, x)
        if i < wh:
            xeval = xx[:2*wh]
            d  = dens[:2*wh]
        elif len(xx) - wh < i:
            xeval = xx[-wh*2:]
            d = dens[-wh*2:]
        else:
            xeval = xx[i-wh:i+wh]
            d  = dens[i-wh:i+wh]
        g = gaussian(xeval, x, scale=scale)
        r.append(np.sum(g))
    return(np.array(r))

def smooth_lowess(yy, xx, xvals=None, it=3, frac=0.66):
    return lowess(yy, xx, xvals=xvals, it=it, frac=frac)

def smooth_result(x, tx, dens, newx):
    #newy = smooth_gaussian(r.tx, r.x, newx, scale=10)
    newy = smooth_lowess(tx, x, newx)
    new_dens = weighted_kde(x, dens, newx, wh=2*len(x), scale=0.01)
    new_dens = new_dens / np.sum(new_dens)
    return newx, newy, new_dens


def find_best_factor(a, b):
    def target(x):
        if x <= 0 or np.isclose(x, 0):
            return np.inf
        aa, bb = regularize(a, b*x)
        try:
            xx, tx, dens = no_kde_algorithm(aa, bb)
        except ZeroDivisionError as e:
            print('Error!', x)
            return np.inf
        cost = no_kde_cost(xx, tx, dens)
        return cost

    cost = np.inf
    guess = 1
    for x0 in [1/1000, 1/100, 1/10, 1, 10, 50, 70, 100, 1000]:
        res = scipy.optimize.minimize(target, x0, method="nelder-mead", options={"xatol": 1e-8})
        if res.fun < cost:
            cost = res.fun
            guess = res.x
    return guess, cost

def subsample(x, size=None):
    if size is None:
        size = np.sum(x)
    elif isinstance(size, float):
        size = int(np.sum(x) * size)
    d = np.unique(np.random.choice(np.arange(len(x)),replace=True, p=x/np.sum(x), size=size), return_counts=True)[1]
    d = d[d!= 0]
    d = np.sort(d)[::-1]
    return d

def smooth_gaussian(yy, xx, xvals, scale=1, wh=1000):
    r = []
    for x in xvals:
        i = np.searchsorted(xx, x)
        if i < wh:
            xeval = xx[:2*wh]
            yval  = yy[:2*wh]
        elif len(xx) - wh < i:
            xeval = xx[-wh*2:]
            yval  = yy[-wh*2:]
        else:
            xeval = xx[i-wh:i+wh]
            yval  = yy[i-wh:i+wh]
        g = gaussian(xeval, x, scale=scale)
        v = g * yval / np.sum(g)
        r.append(np.sum(v))
    return(np.array(r))


def _no_kde_algorithm(a, b, expand=False):
    a = np.sort(a)
    b = np.sort(b)
    r = []

    la, lb = len(a), len(b)

    l = lcm(la, lb)
    factor_a = l/la
    factor_b = l/lb

    ia = 0
    ib = 0
    va = a[ia]
    vb = b[ib]
    ra = factor_a
    rb = factor_b

    while ia < la or ib < lb:
        va = a[ia]
        vb = b[ib]

        if ia == la and ib == lb:
            break
        v = min(ra, rb)
        if expand:
            for _ in range(int(v)):
                r.append((va, vb, 1))
        else:
            r.append((va, vb, v))
        ra -= v
        rb -= v
        if ra == 0:
            ia += 1
            ra = factor_a
        if rb == 0:
            ib += 1
            rb = factor_b
    return r

def no_kde_barycenter(xs):
    xs = [np.sort(x) for x in xs]
    result = []
    lengths = [len(x) for x in xs]
    l = lcm(*lengths)
    factors = [l/lenx for lenx in lengths]

    indexes = [0 for x in xs]
    values = [x[i] for i, x in zip(indexes, xs)]
    residuen = [f for f in factors]
    while any(idx < lenx for idx, lenx in zip(indexes, lengths)):
        values = [x[i] for i, x in zip(indexes, xs)]
        v = min(residuen)
        result.append((np.mean(values), v/l))
        residuen = [r-v for r in residuen]
        for i in range(len(residuen)):
            if residuen[i] == 0:
                indexes[i] += 1
                residuen[i] = factors[i]
    return result

def no_kde_weights(xs, ws):
    assert len(xs) == 2
    for x in xs:
        assert all(np.sort(x) == x)
    xs = [x for x in xs]
    result = []
    lengths = [len(x) for x in xs]
    weights = [w/np.sum(w) for w in ws]

    indexes = [0 for x in xs]
    values = [x[i] for i, x in zip(indexes, xs)]
    residuen = [w[0] for w in weights]
    while any(idx < lenx for idx, lenx in zip(indexes, lengths)):
        try:
            values = [x[i] for i, x in zip(indexes, xs)]
        except IndexError:
            break
        v = min(residuen)
        result.append((*values, v))
        residuen = [r-v for r in residuen]

        for i in range(len(residuen)):

            if residuen[i] == 0:
                indexes[i] += 1
                if len(weights[i]) <= indexes[i]:
                    continue
                residuen[i] = weights[i][indexes[i]]

    x, y, v = zip(*result)

    tx = np.array(y) / np.array(x)
    v = np.array(v) / np.sum(v)
    return np.array(x), np.array(tx), np.array(v)

def no_kde_algorithm(a, b, dens_min=1, dens_factor=None):
    r = _no_kde_algorithm(a, b)
    x, y, v = zip(*r)
    tx = np.array(y) / np.array(x)
    v = np.array(v) / np.sum(v)
    return np.array(x), np.array(tx), np.array(v)


def no_kde_cost(x, tx, dens, cells=False):
    assert np.isclose(np.sum(dens), 1)
    if cells:
        return np.sum(np.abs(((tx*x) - x)*dens))
    else:
        return np.sum(np.abs(np.log2(tx)*dens))


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

def emd(s1, s2, counts=300, x_range=None, per_data_axis=None, m=None, mm=None):
    d1 = np.log2(s1)
    d2 = np.log2(s2)

    if x_range is None:
        if m is None:
            m = min(np.min(d1), np.min(d2))
        if mm is None:
            mm = max(np.max(d1), np.max(d2))
        if per_data_axis:
            counts = int((mm - m) * counts)
        x_range = np.linspace(m, mm, counts)
        
    x1, y1 = kde(d1, x_range)
    x2, y2 = kde(d2, x_range)

    y1, y2 = y1 / np.sum(y1), y2 / np.sum(y2)

    M = ot.dist(x1.reshape((len(x1), 1)), x1.reshape((len(x1), 1)))
    Gs = ot.emd(y1, y2, M)
    
    cost = np.sum(M * Gs)
    
    return x_range, y1, y2, M, Gs, cost



def _kde_algorithm(lib1, lib2, counts=150, per_data_axis=False, m=None, mm=None):
    x, y1, y2, M, Gs, cost = emd(lib1, lib2, counts=counts, per_data_axis=per_data_axis, m=m, mm=mm)
    weights = Gs / np.sum(Gs, axis=0).reshape(1, Gs.shape[0])
    T = (x @ weights)
    return x, T, y1

def kde_algorithm(lib1, lib2, dens_factor=30, count=20, dens_min=0, dens_max=np.inf):
    x, tx, dens = _kde_algorithm(np.log2(lib1), np.log2(lib2), count, per_data_axis=True)

    tx = x-tx
    x = np.exp2(x)
    tx = np.exp2(tx)
    dens = dens*dens_factor

    dens[dens > dens_max] = dens_max
    dens[dens < dens_min] = dens_min

    return x, tx, dens

def regularize(l1, l2):
    m = max(np.min(l1), np.min(l2))
    l1 = l1[l1 >= m]
    l2 = l2[l2 >= m]
    return l1, l2

def normalize_libs(lib1, lib2, r):
    N1 = len(lib1)
    N2 = len(lib2)

    if N1 == 0 or N2 == 0:
        return lib1+1, lib2+1, -1

    r_i = N1/N2

    r_corr = r_i / r

    if r_corr < 1:
        lib1 = np.concatenate([lib1, np.array(int(r_corr*N1)*[0])])
    if r_corr > 1:
        lib2 = np.concatenate([lib2, np.array(int(r_corr*N2)*[0])])
    lib1 = np.sort(lib1)
    lib2 = np.sort(lib2)

    return lib1+1, lib2+1, r_corr


def plot_foldchange(x, tx, dens, color="blue", line=False, p=None, **kwargs):
    if p is None:
        if "frame_height" not in kwargs:
            kwargs["frame_height"] = 150
        if "frame_width" not in kwargs:
            kwargs["frame_width"] = 150
        p = figure(**kwargs)

    tx[np.isnan(tx)] = 0

    if line:
        p.line(x=x, y=tx, color=color)
    else:
        x = np.hstack((x, x[::-1]))
        tx = np.hstack((tx+dens, (tx-dens)[::-1]))
        p.patch(x=x, y=tx, color=color, line_color=None)
    return p


def plot_foldchange_no_kde(x, tx, dens, color="#abcdef", p=None, line=None, y_range=None, x_range=None, **kwargs):
    if p is None:
        p = figure(**kwargs, x_range=x_range, y_range=y_range)

    if y_range is None:
        y_range = np.min(tx), np.max(tx)
    if x_range is None:
        x_range = np.min(x), np.max(x)
    cvs = ds.Canvas(plot_width=100, plot_height=250, x_range=x_range, y_range=y_range)
    mask = ~(np.isinf(x) | np.isnan(x) | np.isinf(tx) | np.isnan(tx))
    x = x[mask]
    tx = tx[mask]
    if len(x) > 0:
        agg = cvs.points(pd.DataFrame({"x": x, "tx": tx}), "x", "tx", agg=ds.count())
        image = tf.shade(agg, cmap=[color], how="log")
        p.image_rgba(image=[image.values], x=[x_range[0]], y=[y_range[0]], dw=[x_range[1]-x_range[0]], dh=[y_range[1]-y_range[0]])
    return p

def plot_foldchange_no_kde_smooth(x, tx, dens, color="#abcdef", p=None, line=None, y_range=None, x_range=None, width=0.1, line_color=None, area_alpha=0.3, line_alpha=1.0, scale=1, median_x=None, **kwargs):

    if line_color is None:
        line_color = color

    if median_x is None:
        median_x = np.median(x)
    if y_range is None:
        y_range = np.min(tx), np.max(tx)
    if x_range is None:
        x_range = np.min(x), np.max(x)

    if p is None:
        p = figure(**kwargs, x_range=x_range, y_range=y_range)

    mask = ~(np.isinf(x) | np.isnan(x) | np.isinf(tx) | np.isnan(tx))
    x = x[mask]
    tx = tx[mask]

    p.varea(x=x, y1=tx+dens, y2=tx-dens, color=color, alpha=area_alpha)
    p.line(x, tx, color=line_color, line_dash="dotted", line_alpha=line_alpha)

    median_idx = np.searchsorted(x, median_x)

    median = x[median_idx]
    median_y = tx[median_idx]
    median_new_dens = dens[median_idx]
    p.circle(x=[median], y=[median_y], line_color=line_color, fill_color=None)
    return p

