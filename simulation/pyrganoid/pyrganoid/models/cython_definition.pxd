import numpy as np
cimport numpy as np
from cython_gsl cimport gsl_rng

cdef class CellType:
    cdef readonly str name
    cdef readonly long start_amount
    cdef readonly float timestep
    cdef public long _id

    cdef readonly list connections

cdef class Connection:
    cdef readonly CellType source
    cdef readonly list sinks
    cdef readonly list amounts
    cdef readonly unsigned int length
    cdef readonly float timestep
    cdef readonly float _rate
    cdef readonly np.ndarray _outvector
    cdef readonly double[:] _rate_rates
    cdef readonly double[:] _rate_timepoints
    cdef long _i
    cdef float _last_t

    cdef long[:] _get_outvector(self, gsl_rng* rng)
    cpdef void reset(self)

    cpdef float rate(self, float t, long total=*)
    cdef float get_rate(self, float t, long total=*)
    cpdef np.ndarray[long, ndim=2] yule_tick(self, np.ndarray[long, ndim=2] cells)



