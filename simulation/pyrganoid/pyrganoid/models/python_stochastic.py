import math
import numpy as np
import numpy.random as random
from typing import List, Union, Callable

from .python_definition import CellType

__all__ = ["handle_type", "set_rng", "init_types", "simulate_lineage_continious"]


def set_rng(value):
    raise NotImplementedError


def simulate_organoid_continious_inner(
    cells, total_time, connections, return_timesteps
):
    print("Using python version")
    returns = []
    time = 0.0
    while time <= total_time:
        if return_timesteps[0] < time:
            returns.append(cells.copy())
            return_timesteps.pop(0)

        # recalculate all rates
        new_connections = []
        total_rate = 0.0
        for type, con, rate in connections:
            total_source_cells = np.sum(cells[type._id, :])
            new_rate = con.rate(time, total_source_cells) * total_source_cells
            total_rate += new_rate
            new_connections.append((type, con, new_rate))
        connections = new_connections

        # calculate when the next event will happen
        next_time = np.random.exponential(1.0 / total_rate)
        connection_p = np.random.rand(1) * total_rate

        cummulative_rate = 0.0
        for type, con, rate in connections:
            cummulative_rate += rate
            if connection_p < cummulative_rate:
                break
        else:
            raise Exception("Could not find a next event, something is wrong")

        con.yule_tick(cells)
        time += next_time

    return returns


def init_types(types: List[CellType], types_to_id):
    for type in types:
        type._id = types_to_id[type]
        for con in type.connections:
            con.init_outvector(types_to_id)


def simulate_lineage_continious_inner(
    types: List[CellType],
    total_time: float,
    return_timesteps: np.ndarray,
    ids_of_source_rates: np.ndarray,
    cells: np.ndarray,
    amount: int,
):
    for i in range(amount):
        j = 0
        time = 0.0
        connections = [[type, con] for type in types for con in type.connections]
        while time <= total_time:
            while time > return_timesteps[j]:
                cells[i, j + 1] = cells[i, j]
                j += 1
            rates = np.array(
                [type_and_con[1].rate(time) for type_and_con in connections]
            )

            next_events = np.random.exponential(
                1 / (rates * cells[i, j][ids_of_source_rates]), 3
            )
            # -np.log(np.random.rand(3)) / (rates * cells[ids_of_source_rates])
            next_event = np.argmin(next_events)
            cells[i, j, :] = cells[i, j, :] + connections[next_event][1]._outvector
            time += next_events[next_event]
