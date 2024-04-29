# cython: language_level=3
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1
import numpy as np
cimport numpy as np
np.import_array()
cimport cython
from libc.stdlib cimport malloc, free

import numpy.random as random

from .cython_definition cimport CellType, Connection

from libc.math cimport exp, isinf, sqrt
from libc.time cimport time

from cython_gsl cimport gsl_ran_binomial, gsl_ran_gaussian, gsl_rng, gsl_rng_alloc, gsl_rng_taus2, gsl_rng_set, gsl_ran_exponential, gsl_rng_uniform

__all__ = ["handle_type", "set_rng", "simulate_organoid_continious_inner"] 

cdef gsl_rng* rng = gsl_rng_alloc(gsl_rng_taus2)
gsl_rng_set(rng, 314159265)

cpdef void set_rng(long value):
    gsl_rng_set(rng, value)

cdef class TypeConRate:
    cdef public CellType type
    cdef public Connection con
    cdef public float rate

    def __init__(self, type, con, rate):
        self.type = type
        self.con = con
        self.rate = rate

    def __repr__(self):
        return f"RateCon({repr(self.con)}, {self.rate})"

cpdef list simulate_organoid_continious_inner(np.ndarray[long, ndim=2] cells, float total_time, list p_connections, list return_timesteps, double[:] rates_change_at):
    print("Using cython version")
    cdef long i = 0
    cdef double time = 0.0
    cdef float total_rate = 0.0
    cdef list connections = []
    cdef int connections_len = len(p_connections)
    cdef TypeConRate type_con_rate

    cdef list returns = []

    for type, con, rate in p_connections:
        t = TypeConRate()
        t.type = type
        t.con = con
        t.rate = rate
        connections.append(t)

    cdef long total_source_cells
    cdef float new_rate 
    cdef double next_time
    cdef float connection_p
    cdef float cummulative_rate

    while time <= total_time:

        if return_timesteps[0] < time:
            print("Processing " + str(time))
            print("Currently at " + str(np.sum(cells)) + " many cells")
            returns.append(cells.copy())
            return_timesteps.pop(0)

        # recalculate all rates
        total_rate = 0.0
        for i in range(connections_len):
            total_source_cells = np.sum(cells[connections[i].type._id, :])
            new_rate = connections[i].con.rate(time, total_source_cells) * total_source_cells
            total_rate += new_rate
            connections[i].rate = new_rate

        # calculate when the next event will happen
        next_time = gsl_ran_exponential(rng, 1.0/total_rate)
        #next_time = np.random.exponential(1.0/total_rate)
        connection_p = gsl_rng_uniform(rng) * total_rate
        #connection_p = np.random.rand(1) * total_rate

        # find which event happens
        cummulative_rate = 0.0
        for i in range(connections_len):
            type_con_rate = connections[i]
            cummulative_rate += type_con_rate.rate
            if connection_p < cummulative_rate:
                break
        else:
            raise Exception("Could not find a next event, something is wrong")

        # process the event
        type_con_rate.con.yule_tick(cells)

        # advance time
        time += next_time

    return returns

@cython.initializedcheck(True)
@cython.boundscheck(True)
@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.profile(True)
cdef long long simulate_lineage_continious_inner_inner_binomial(list types, float total_time, double[:] return_timesteps, long[:] ids_of_source_rates, long long[:, :, :] cells, long i, long long max_cell_gain, double[:] rates_change_at):
        cdef long j =0
        cdef double time = 0.0
        cdef np.ndarray[double, ndim=1] rates
        cdef np.ndarray[double, ndim=1] next_events
        cdef int len_types = len(types)
        cdef int next_event = 0
        cdef double next_time = 0.0
        cdef long long events = 0

        cdef list connections = [TypeConRate(type, con, 0.0) for type in types for con in type.connections]
        cdef int number_of_connections = len(connections)
        cdef float t
        cdef float r, rr
        cdef long[:] out
        cdef int _id
        cdef Connection con

        for type in types:
            for con in type.connections:
                con.reset()

        cdef long long total_cell_gain = 0

        cdef TypeConRate con_n
        cdef bint recalculate_needed = True
        cdef long rcai = 0

        while time <= total_time:

            while recalculate_needed:

                # update rates and find biggest rate
                r = 0
                rr = 0
                for n in range(number_of_connections):
                    con_n = connections[n]
                    _id = con_n.type._id
                    rr = con_n.con.rate(time)
                    if rr > r:
                        r = rr
                    con_n.rate = rr

                # This should guarantee that the chance of a cell dividing twice in the timespan of next_time is less than 0.01
                next_time =  (1-0.99)/r

                if time+next_time < rates_change_at[rcai] or isinf(rates_change_at[rcai]):
                    recalculate_needed = False
                else:
                    next_time = rates_change_at[rcai] - time
                    time = rates_change_at[rcai]
                    rcai += 1
            recalculate_needed = True

            #with cython.boundscheck(False):

            if (isinf(next_time)):
                time = rates_change_at[rcai]
            else:
                if time + next_time == time:
                    return -2
                time += next_time

            while return_timesteps[j] < time:
                #with cython.boundscheck(False):
                    #with cython.wraparound(False):
                for k in range(len_types):
                    cells[i, j+1, k] = cells[i, j, k]
                j += 1

            for n in range(number_of_connections):
                con_n = connections[n]
                con_n.rate
                _id = con_n.type._id
                out = con_n.con._get_outvector(rng)
                if cells[i, j, _id] > 1e8:
                    events = <long long> (gsl_ran_gaussian(rng, sqrt((con_n.rate*next_time)*(1-(con_n.rate*next_time))*cells[i, j, _id])) + (con_n.rate*next_time)*cells[i, j, _id])
                else:
                    events = gsl_ran_binomial(rng, con_n.rate*next_time, cells[i, j, _id])
                for k in range(len_types):
                    cells[i,j,k] = cells[i,j,k] + out[k] * events
        return total_cell_gain

@cython.initializedcheck(True)
@cython.boundscheck(True)
@cython.wraparound(False)
#@cython.cdivision(True)
#@cython.profile(True)
cdef long long simulate_lineage_continious_inner_inner(list types, float total_time, double[:] return_timesteps, long[:] ids_of_source_rates, long long[:, :, :] cells, long i, long long max_cell_gain, double[:] rates_change_at):
        cdef long j = 0
        cdef double time = 0.0
        cdef np.ndarray[double, ndim=1] rates
        cdef np.ndarray[double, ndim=1] next_events
        cdef int len_types = len(types)
        cdef int next_event = 0
        cdef double next_time = 0.0

        cdef list connections = [TypeConRate(type, con, 0.0) for type in types for con in type.connections]
        cdef int number_of_connections = len(connections)
        cdef float t
        cdef float r
        cdef long[:] out
        cdef int _id
        cdef Connection con

        for type in types:
            for con in type.connections:
                con.reset()

        cdef long long total_cell_gain = 0

        cdef TypeConRate con_n
        cdef bint recalculate_needed = True
        cdef long rcai = 0

        while time <= total_time and total_cell_gain < max_cell_gain:

            while recalculate_needed:

                for n in range(number_of_connections):
                    con_n = connections[n]
                    _id = con_n.type._id
                    con_n.rate = con_n.con.rate(time) * cells[i, j, _id]

                next_event = 0
                r = connections[0].rate
                if r == 0:
                    next_time = float("inf")
                else:
                    next_time = gsl_ran_exponential(rng, 1.0/r)
                for n in range(1, number_of_connections):
                    r = connections[n].rate
                    if r == 0:
                        t = float("inf")
                    else:
                        t = gsl_ran_exponential(rng, 1.0/r)
                    if t < next_time:
                        next_time = t
                        next_event = n

                if time+next_time < rates_change_at[rcai] or isinf(rates_change_at[rcai]):
                    recalculate_needed = False
                else:
                    time = rates_change_at[rcai]
                    rcai += 1
            recalculate_needed = True

            #with cython.boundscheck(False):

            if (isinf(next_time)):
                time = rates_change_at[rcai]
            else:
                if time + next_time == time:
                    return -2
                time += next_time

            while return_timesteps[j] < time:
                #with cython.boundscheck(False):
                    #with cython.wraparound(False):
                for k in range(len_types):
                    cells[i, j+1, k] = cells[i, j, k]
                j += 1

            con = connections[next_event].con
            out = con._get_outvector(rng)
            for k in range(len_types):
                total_cell_gain = total_cell_gain + out[k]
                cells[i,j,k] = cells[i,j,k] + out[k]
        return total_cell_gain

cpdef long long simulate_lineage_continious_inner(list types, float total_time, double[:] return_timesteps, long[:] ids_of_source_rates, long long[:, :, :] cells, long amount, long long max_cell_gain, double[:] rates_change_at, bint use_binomial):
    for i in range(amount):
        if use_binomial:
            total_cell_gain = simulate_lineage_continious_inner_inner_binomial(types, total_time, return_timesteps, ids_of_source_rates, cells, i, max_cell_gain, rates_change_at)
        else:
            total_cell_gain = simulate_lineage_continious_inner_inner(types, total_time, return_timesteps, ids_of_source_rates, cells, i, max_cell_gain, rates_change_at)
        #max_cell_gain = max_cell_gain - total_cell_gain
    return max_cell_gain

# DIRTY HACK FOR LINE PROFILING https://github.com/pyutils/line_profiler/issues/13
def activate_line_profiling():
    pass
