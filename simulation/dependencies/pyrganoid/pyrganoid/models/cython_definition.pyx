"""
Contains the classes to define models
"""
# cython: language_level=3
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1

__all__ = ["CellType", "Connection", "Delayed", "Delayed_Increasing", "Capacity", "DelayedCapacity", "Moran", "Between"]
from cython_gsl cimport gsl_ran_binomial, gsl_rng

from typing import List, Union
import numpy as np
cimport numpy as np

cdef class CellType:
    """
    Defines types of cells which can undergo events. The name *must* be unique.

    CellType(name, start_amount)

    This is the main class to define models. First define which types of cells
    your model should contain with this class. Then define the events between
    them via the `connect` method. As a last step use either
    pyrganoid.models.deterministic or pyrganoid.models.stochastic to do either a
    deterministic or stochastic simulation. For examples see the README.md

    This class only stores the information needed to run the simulations. The
    more interesting part is in the models themselves.
    """

    def __init__(self, name, start_amount, timestep=0.01):
        self.name = name
        self.start_amount = start_amount
        self.timestep = timestep
        self.connections = list()

    def __str__(self):
        return f"CellType({self.name}, start_amount={self.start_amount})"

    def __repr__(self):
        return f"CellType({self.name}, start_amount={self.start_amount})"
    
    def connect(self, others: Union['CellType', List['CellType']], 
            amounts: Union[int, List[int]], 
            connection=None,
            *args,
            **kwargs
            ):
        """
        Link to another cell type via an event

        Args:
            others: List of produced cell types by this event. Lists with length
                    1 can be written as only the cell type.
            amounts: List of produced offspring, in the same order as `others`.
                     Listes of length 1 can again be written simply as an
                     integer.
            name: Name of the event, only used for debugging
            connection: Class of connection, default Connection
            *args/**kwargs: will be passed to the connection class, source and 
                     timestep are provided automatically

        Defines an event which produces another cell. Note that if the cell
        itself does not die/vanish/change it also needs to be defined as output:

            # "wrong", the cell_type cell will vanish after the event, and only
            # one other_type will be there
            cell_type.connect(other_type, "produce one other_type", 1, rate=1)
            
            # "right" if you want one other_type cell additionally to the
            # cell_type cell after the event
            cell_type.connect([cell_type, other_type], "produce one other_type", [1, 1], rate=1)
        """

        if isinstance(others, CellType):
            others = [others]
        if not isinstance(amounts,  list):
            amounts = [amounts]
        assert(len(others) == len(amounts))
        if connection is None:
            connection = Connection

        con = connection(self, self.timestep, others, amounts, *args, **kwargs)
        self.connections.append(con)
        
    def __hash__(self):
        """
        Custom hash for interop with jug

        Jug needs to pickle python objects if they are used as keys in
        dictionaries. If the hash changes cell types can't be identified between
        processes and runs.
        """
        return hash(self.name)

cdef class Connection:

    def __init__(self, source, timestep, sinks, amounts, rate):
        self.source = source
        self.sinks = sinks
        self.amounts = amounts
        self.timestep = timestep
        self.length = len(amounts)
        self._last_t = -1.0

        if isinstance(rate, float) or isinstance(rate, int):
            if rate < 0:
                rate = 0
            rate = [(rate, np.inf)]

        rate_rates, rate_times = list(zip(*rate))
        rate_rates = list(rate_rates)
        rate_times = list(rate_times)
        rate_times.append(np.inf)
        rate_rates.append(rate_rates[-1])

        self._rate_rates =  np.array(rate_rates, dtype=np.double)
        self._rate_timepoints =  np.array(rate_times, dtype=np.double)
        self._outvector = np.zeros([1], dtype=np.int64)
        self._i = 0
        self._rate = self._rate_rates[0]

    def __str__(self):
        return f"Connection({self.source} -> {self.sinks}, {self.amounts})"

    def __repr__(self):
        return f"Connection({self.source} -> {self.sinks}, {self.amounts})"

    def init_outvector(self, types_to_id):
        self._outvector = np.zeros(len(types_to_id), dtype=np.int64)
        for sink, amount in zip(self.sinks, self.amounts):
            self._outvector[types_to_id[sink]] += amount
        self._outvector[types_to_id[self.source]] -= 1

    def init_deterministic(self):
        pass
    
    cpdef float rate(self, float t, long total=0):
        if self._last_t > t:
            self.reset()
        while self._rate_timepoints[self._i] <= t:
            self._i += 1
        self._last_t = t
        return self._rate_rates[self._i]

    cpdef void reset(self):
        self._i = 0

    cdef float get_rate(self, float t, long total=0):
        if self._last_t > t:
            self.reset()
        while self._rate_timepoints[self._i] <= t:
            self._i += 1
        self._last_t = t
        return self._rate_rates[self._i]

    cdef long[:] _get_outvector(self, gsl_rng* rng):
        return self._outvector

    cpdef np.ndarray[long, ndim=2] yule_tick(self, np.ndarray[long, ndim=2] cells):
        c = cells[self.source._id, :]
        chosen = <long> np.random.choice(range(len(cells[0, :])), p=c/np.sum(c))
        cells[:, chosen] += self._outvector
        return cells

cdef class Delayed(Connection):
    """
    return 0 until time, then return rate.

    Jug doesn't handle lambdas or anonymous functions, thus we need a "proper
    closure" implemented as a class. Used with the `delayed` function.
    """

    cdef readonly float _time

    def __init__(self, source, timestep, sinks, amounts, rate, time):
        Connection.__init__(self, source, timestep, sinks, amounts, rate)
        self._time = time

    cpdef float rate(self, float t, long total=0):
        if t < (self._time):
            return 0.0
        else:
            return self._rate

cdef class Between(Connection):
    """
    Returns 0 until time1, then return rate until time2, then return 0.
    """

    cdef readonly float _time1
    cdef readonly float _time2

    def __init__(self, source, timestep, sinks, amounts, rate, time1, time2):
        Connection.__init__(self, source, timestep, sinks, amounts, rate)
        self._time1 = time1
        self._time2 = time2

    cpdef float rate(self, float t, long total=0):
        if t < (self._time1):
            return 0.0
        elif t < (self._time2):
            return self._rate
        else:
            return 0.0

cdef class Until(Connection):
    """
    return rate until time, then return 0.

    Jug doesn't handle lambdas or anonymous functions, thus we need a "proper
    closure" implemented as a class. Used with the `delayed` function.
    """

    cdef readonly float _time

    def __init__(self, source, timestep, sinks, amounts, rate, time):
        Connection.__init__(self, source, timestep, sinks, amounts, rate)
        self._time = time

    cpdef float rate(self, float t, long total=0):
        if t < (self._time):
            return self._rate
        else:
            return 0.0

cdef class Delayed_Increasing(Delayed):
    """
    return 0 until time, then return rate * dt.

    Jug doesn't handle lambdas or anonymous functions, thus we need a "proper
    closure" implemented as a class. Used with the `delayed` function.
    """

    cpdef float rate(self, float t, long total=0):
        if t < (self._time):
            return 0.0
        else:
            return self._rate * (t-self._time)

cdef class Capacity(Connection):
    """
    Return 0 until capacity is reached, then rate * (1 - capacity/total)
    """

    cdef readonly long _capacity
    def __init__(self, source, timestep, sinks, amounts, rate, capacity):
        Connection.__init__(self, source, timestep, sinks, amounts, rate)
        self._capacity = capacity

    cpdef float rate(self, float t, long total=0):
        if total > self._capacity:
            return 0.0
        else:
            return self._rate * (1 - total/self._capacity)

cdef class DelayedCapacity(Capacity):
    """
    Return 0 until time, then return 0 if capacity > total, then return rate * (1 - capacity/total)
    """

    cdef readonly float _time
    def __init__(self, source, timestep, sinks, amounts, rate, capacity, time):
        Capacity.__init__(self, source, timestep, sinks, amounts, rate, capacity)
        self._time = time

    cpdef float rate(self, float t, long total=0):
        if t < self._time:
            return 0.0
        if total > self._capacity:
            return 0.0
        return self._rate * (1 - total/self._capacity)

cdef class Moran(Connection):
    cdef readonly list sinks2
    cdef readonly list amounts2
    cdef readonly np.ndarray _outvector2

    def __init__(self, source, timestep, sinks, amounts, rate, sinks2, amounts2):
        Connection.__init__(self, source, timestep, sinks, amounts, rate)
        self.sinks2 = sinks2
        self.amounts2 = amounts2

    def init_outvector(self, types_to_id):
        self._outvector = np.zeros(len(types_to_id), dtype=np.int64)
        for sink, amount in zip(self.sinks, self.amounts):
            self._outvector[types_to_id[sink]] += amount

        self._outvector2 = np.zeros(len(types_to_id), dtype=np.int64)
        for sink, amount in zip(self.sinks2, self.amounts2):
            self._outvector2[types_to_id[sink]] += amount

    def init_deterministic(self):
        self.sinks = [*self.sinks, *self.sinks2]
        self.amounts = [*self.amounts, *self.amounts2]

    cpdef np.ndarray[long, ndim=2] yule_tick(self, np.ndarray[long, ndim=2] cells):
        p = cells[self.source._id, :]
        p = p / np.sum(p)
        chosen = np.random.choice(range(len(cells[0, :])), p=p)
        chosen2 = np.random.choice(range(len(cells[0, :])), p=p)
        cells[:, chosen] += self._outvector
        cells[:, chosen2] += self._outvector2

cdef class Binomial(Connection):
    """
    Represents variable offspring which is binomially distributed
    """

    cdef readonly double _p
    cdef readonly unsigned int _n
    cdef readonly list _static_sinks
    cdef readonly list _static_amounts
    cdef readonly np.ndarray _static_outvector

    def __init__(self, source, timestep, sinks, amounts, rate, n, p, static_sinks=[], static_amounts=[]):
        Connection.__init__(self, source, timestep, sinks, amounts, rate)
        self._n = n
        self._p = p
        self._static_sinks = static_sinks
        self._static_amounts = static_amounts

    def init_outvector(self, types_to_id):
        self._outvector = np.zeros(len(types_to_id), dtype=np.int64)
        for sink, amount in zip(self.sinks, self.amounts):
            self._outvector[types_to_id[sink]] += amount

        self._static_outvector = np.zeros(len(types_to_id), dtype=np.int64)
        for sink, amount in zip(self._static_sinks, self._static_amounts):
            self._static_outvector[types_to_id[sink]] += amount
        self._static_outvector[types_to_id[self.source]] -= 1

    def init_deterministic(self):
        self.sinks = [*self.sinks, *self._static_sinks]
        self.amounts = [*list(np.array(self.amounts)*self._n * self._p), *self._static_amounts]

    cdef long[:] _get_outvector(self, gsl_rng* rng):
        cdef long var = gsl_ran_binomial(rng, self._p, self._n)
        return self._outvector * var + self._static_outvector
