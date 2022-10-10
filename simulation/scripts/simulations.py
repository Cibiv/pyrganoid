import numpy as np
import numpy.random

from pyrganoid.models.definition import CellType
from pyrganoid.models.stochastic import simulate_lineage_continious

def reads_per_million(*args):
    s = sum(np.sum(x) for x in args)
    if len(args) > 1:
        return [1000000 * (x / s) for x in args]
    else:
        return 1000000 * (args[0] / s)



def generate_model_san(a_rate):

    S = CellType('S', 1)
    A = CellType('A', 0)
    N = CellType('N', 0)

    S.connect(S, 2, rate=1.5)
    S.connect(A, 1, rate=1.5)
    A.connect([A, N], [1, 1], rate=a_rate)

    days = [0, 1, 3, 6, 9, 11, 13, 21, 35, 40]
    out = simulate_lineage_continious([S, A, N], 40, days, 10000, 1e18)

    dats = {int(out['time'][i]): reads_per_million(np.sort(np.sum(out['lineages'][:, i, :], axis=1))[::-1]) for i in range(len(days))}

    ar = out['lineages']
    for i in range(len(days)):
        ar[:, i, :] = reads_per_million(ar[:, i, :])

    return dats, out

def generate_model_san2(a_rate):

    S = CellType('S', 1)
    A = CellType('A', 0)
    N = CellType('N', 0)

    S.connect(S, 2, rate=1.5)
    S.connect(A, 1, rate=1.0)
    A.connect([A, N], [1, 1], rate=a_rate)

    days = [0, 1, 3, 6, 9, 11, 13, 21, 35, 40]
    out = simulate_lineage_continious([S, A, N], 40, days, 10000, 1e18)

    dats = {int(out['time'][i]): reads_per_million(np.sort(np.sum(out['lineages'][:, i, :], axis=1))[::-1]) for i in range(len(days))}
    for i in range(len(days)):
        out['lineages'][:, i, :] = reads_per_million(out['lineages'][:, i, :])
    return dats, out

def generate_model_san3(a_rate):

    S = CellType('S', 1)
    A = CellType('A', 0)
    N = CellType('N', 0)

    S.connect(S, 2, rate=1.5)
    S.connect(A, 1, rate=2.0)
    A.connect([A, N], [1, 1], rate=a_rate)

    days = [0, 1, 3, 6, 9, 11, 13, 21, 35, 40]
    out = simulate_lineage_continious([S, A, N], 40, days, 10000, 1e18)

    dats = {int(out['time'][i]): reads_per_million(np.sort(np.sum(out['lineages'][:, i, :], axis=1))[::-1]) for i in range(len(days))}
    for i in range(len(days)):
        out['lineages'][:, i, :] = reads_per_million(out['lineages'][:, i, :])
    return dats, out


def simulate_lineage(days, cutoffpoints, p=0.1):

    timepoints = sorted(list(days) + list(cutoffpoints))

    time = 0
    cells = 1
    out = {}
    for timepoint in timepoints:
        S = CellType('S', cells)
        S.connect(S, 2, rate=1.5)
        length = timepoint - time
        time = time + length
        output = simulate_lineage_continious([S], length, [length], 1, 1e18)
        cells = output['lineages'][0, 0, 0]
        if timepoint in days:
            out[timepoint] = cells
        if timepoint in cutoffpoints:
            cells = np.random.binomial(cells, p)
    return out


def generate_model_s(p=0.1):
    lineages = 10000

    days = [0, 1, 3, 6, 9, 11, 13, 21, 35, 40]
    cutoffpoints = [7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42]
    out = []
    for i in range(lineages):
        data = simulate_lineage(days, cutoffpoints, p=p)
        out.append(data)

    dats = {day: np.array([replicate[day] for replicate in out]) for day in days}

    dats = {day: reads_per_million(dats[day])for day in days}
    return dats, False


def generate_model_a():
    A = CellType('A', 1)
    N = CellType('N', 0)
    A.connect([A, N], [1, 1], rate=0.7)

    days = [0, 1, 3, 6, 9, 11, 13, 21, 35, 40]
    out = simulate_lineage_continious([A, N], 40, days, 10000, 1e18)

    dats = {int(out['time'][i]): reads_per_million(np.sort(np.sum(out['lineages'][:, i, :], axis=1))[::-1]) for i in range(len(days))}
    for i in range(len(days)):
        out['lineages'][:, i, :] = reads_per_million(out['lineages'][:, i, :])
    return dats, out

