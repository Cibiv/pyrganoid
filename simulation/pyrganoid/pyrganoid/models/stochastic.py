"""Functions to run a stochastic simulation"""
__all__ = ["start", "repeat", "simulate_organoid", "simulate_organoid_continious"]

import numpy as np
import numpy.random as random
import pandas
from typing import List, Union, Dict
import math
import os

from .definition import CellType

from .definition import CellType, Connection, Delayed, Delayed_Increasing
from .definition import set_rng

USE_PYTHON = os.environ.get("PYRGANOID_USE_PYTHON")
if USE_PYTHON and USE_PYTHON.upper() == "YES":
    from .python_stochastic import (
        simulate_organoid_continious_inner,
        simulate_lineage_continious_inner,
    )
elif USE_PYTHON and USE_PYTHON.upper() == "NO":
    from .cython_stochastic import (
        simulate_organoid_continious_inner,
        simulate_lineage_continious_inner,
    )
# Use cython per default if possible
else:
    try:
        from .cython_stochastic import (
            simulate_organoid_continious_inner,
            simulate_lineage_continious_inner,
        )
    except ModuleNotFoundError:
        from .python_stochastic import (
            simulate_organoid_continious_inner,
            simulate_lineage_continious_inner,
        )

        print(
            "Couldn't find the compiled cython modules, fallback to python. This might be slow!"
        )
from .python_stochastic import init_types


def simulate_organoid_continious(
    types: List[CellType],
    starting_cells: int,
    total_time: int,
    return_timesteps: List[int],
):
    cells = np.zeros((len(types), starting_cells), dtype=np.int64)
    connections = [[type, con, 0] for type in types for con in type.connections]

    time = 0.0
    returns: np.ndarray[np.int64] = []

    types_to_id = {type: i for i, type in enumerate(types)}

    for type in types:
        type._id = types_to_id[type]
        print(type)
        print(type._id)
        cells[type._id, :] = type.start_amount

    init_types(types, types_to_id)

    print(types)
    for type in types:
        for con in type.connections:
            print(con)
            print(con._outvector)

    # Copy the list because we are going to mutate it
    return_timesteps = sorted([x for x in return_timesteps])

    r = simulate_organoid_continious_inner(
        cells, total_time, connections, return_timesteps
    )
    returns = [*r, *returns]

    returns.append(cells.copy())
    return returns


def merge_and_sample(types, datas, samplepoints, timestep):
    """Merge multiple runs into a pandas dataframe and throw away unneeded data"""
    numbers = len(datas)
    index = [x.name for x in types]
    totals = pandas.DataFrame(index=list(range(numbers)), columns=samplepoints)
    df = pandas.DataFrame(index=index, columns=samplepoints).fillna(0)
    for i, data in enumerate(datas):
        ddf = pandas.DataFrame(data)
        for sample in samplepoints:
            data = ddf.T[(sample) / timestep]
            for ind in index:
                df[sample][ind] = data[ind]
        totals.at[i] = df.sum()

    return totals


def repeat(types, numbers=20, samples=[0, 40], timestep=0.001, use_jug=False):
    """Repeat a simulation for given number of times, and only return specific samples

    Args:
        types: List of celltypes to simulate
        numbers: how often to repeat the simulation
        samples: which timepoints to sample
        timestep: timestep of simulation, see `start()`
        use_jug: use jug for multiprocessing

    Returns a pandas dataframe with created only with the data from the
            timepoints specified.
    """

    datas = []
    for i in range(numbers):
        if use_jug:
            import jug

            data = jug.Task(start, types, 41, timestep, _jug=i)
        else:
            data = start(types, 41, timestep)
        datas.append(data)

    if use_jug:
        import time

        print("You need to run some jug workers, otherwise this will never finish")
        if not jug.is_jug_running():
            for task in datas:
                if task.can_load():
                    continue
                task.run()
        datas = [task.value() for task in datas]
        if jug.is_jug_running():
            import sys

            sys.exit(0)

    index = [x.name for x in types]
    totals = pandas.DataFrame(index=list(range(numbers)), columns=samples)
    df = pandas.DataFrame(index=index, columns=samples).fillna(0)
    for i, data in enumerate(datas):
        ddf = pandas.DataFrame(data)
        for sample in samples:
            data = ddf.T[(sample) / timestep]
            for ind in index:
                df[sample][ind] = data[ind]
        totals.at[i] = df.sum()

    return totals


def simulate_lineage_continious(
    types: List[CellType], total_time: int, return_timesteps: List[int], amount: int, max_cell_gain: int, use_binomial: bool = True
):
    total_time = max(return_timesteps)
    cells = np.zeros((amount, len(return_timesteps) + 2, len(types)), dtype=np.int64)
    types_to_id = {type: i for i, type in enumerate(types)}
    return_timesteps_simulation = np.array(
        [-1] + list(return_timesteps) + [np.inf], dtype=np.float64
    )
    return_timesteps_real = np.array(list(return_timesteps), dtype=np.float64)

    init_types(types, types_to_id)

    for type in types:
        cells[:, 0, type._id] = type.start_amount

    connections = [[type, con] for type in types for con in type.connections]
    ids_of_source_rates = np.array(
        [types_to_id[type_and_con[0]] for type_and_con in connections]
    )

    rates_change_at = set()
    for _, con in connections:
        for t in con._rate_timepoints:
            rates_change_at.add(t)
    timepoints = np.array(sorted(list(rates_change_at)) + [np.inf])

    remaining_max_cell_gain = simulate_lineage_continious_inner(
        types,
        total_time,
        return_timesteps_simulation,
        ids_of_source_rates,
        cells,
        amount,
        max_cell_gain,
        timepoints,
        use_binomial
    )

    if remaining_max_cell_gain <= 0:
        print("Too many cells, aborting")

    r_dict = {}
    r_dict["time"] = return_timesteps_real
    r_dict["lineages"] = cells[:, 1:-1, :]
    for type in types:
        r_dict[type] = np.sum(cells[:, 1:-1, types_to_id[type]], axis=0)
    r_dict["celltypes"] = types

    return r_dict
