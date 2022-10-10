# cython: language_level=3
from .cython_definition cimport CellType, Connection
cimport numpy as np
import numpy as np

cpdef list simulate_organoid_continious_inner(np.ndarray[long, ndim=2] cells, float total_time, list p_connections, list return_timesteps, double[:] rates_change_at)


cpdef long long simulate_lineage_continious_inner(list types, float total_time, double[:] return_timesteps, long[:] ids_of_source_rates, long long[:, :, :] cells, long amount, long long max_cell_gain, double[:] rates_change_at, bint use_binomial)

cdef long long simulate_lineage_continious_inner_inner(list types, float total_time, double[:] return_timesteps, long[:] ids_of_source_rates, long long[:, :, :] cells, long i, long long max_cell_gain, double[:] rates_change_at)

cdef long long simulate_lineage_continious_inner_inner_binomial(list types, float total_time, double[:] return_timesteps, long[:] ids_of_source_rates, long long[:, :, :] cells, long i, long long max_cell_gain, double[:] rates_change_at)
