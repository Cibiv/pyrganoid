from typing import Callable, Tuple, Union, List, Dict
import numpy as np
import scipy.integrate as integrate
import scipy.optimize

from .definition import CellType

TimeDependentParameter = Callable[[float], float]
Parameter = Union[float, TimeDependentParameter]


def start_deterministic(
    types: List[CellType],
    total_time: int,
    amount_multiplier: int = 1,
    samples: int = 10000,
) -> Dict[str, np.array]:
    """
    Do a deterministic simulation.

    Args:
        types: The list of cell types that should be simulated
        total_time: How much time we will simulate
        amount_multiplier: The starting amount gets multiplied with this value.
                           In this way you can use the same model definitions
                           as for the stochastic one, but in the stochastic case
                           you still only simulate a single cell lineage.
        samples: Because the differential equations are solved numerically we
                 we need to know how many steps should be taken.

    Returns a dictionary with cell type keys and arrays of cell counts as value.
    Additionally the dictionary contains a "time" key which value is the time for
    each array point.
    """

    init = [type.start_amount * amount_multiplier for type in types]

    in_going: Dict[CellType, List[Tuple[float, float, CellType]]] = {
        type: [] for type in types
    }
    for type in types:
        for connection in type.connections:
            connection.init_deterministic()
            for amount, sink in zip(connection.amounts, connection.sinks):
                in_going[sink].append((connection.rate, amount, type))

    # needed in deriv() to get the amount of source cells
    index = {type: i for i, type in enumerate(types)}

    # scipy needs a function that calculates the derivate to do the ODE integration.
    # scipy calls that function and passes the time and the current value of the function y=f(t)
    def deriv(y, t):
        r = []
        # calculate the derivate for all types
        for current_value, type in zip(y, types):
            v = 0
            # first calculate all the events that lead to an increase of this
            # cell type
            for rate, amount, in_type in in_going[type]:
                try:
                    rate = rate(t, y[index[in_type]])
                except TypeError:
                    pass
                # the derivate is proportional to the amount of source cells,
                # thus the y[index[in_type]] lookup
                v += rate * amount * y[index[in_type]]
            # now substract each events which "consumes" one cell type
            # note that "self maintaining" events had just added 1 in the step
            # beforehand although derivate for them should be 0
            for connection in type.connections:
                rate = connection.rate
                try:
                    rate = connection.rate(t, y[index[in_type]])
                except TypeError:
                    pass
                v -= rate * current_value
            r.append(v)
        return r

    x = np.linspace(0, total_time, samples)
    y = integrate.odeint(deriv, init, x)

    return {
        "time": x,
        **{type: data for type, data in zip(types, y.T)},
        "celltypes": types,
    }


def deterministic_to_end_cell_count(cell_types, result):
    total = 0.0
    for type in cell_types:
        total += result[type][-1]
    return total
