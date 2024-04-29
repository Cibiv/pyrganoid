__all__ = [
    "CellType",
    "Connection",
    "Delayed",
    "Delayed_Increasing",
    "Until",
    "Capacity",
    "DelayedCapacity",
    "set_rng",
    "Moran",
    "Binomial",
    "Between",
]

try:
    from .cython_definition import (
        CellType,
        Connection,
        Delayed,
        Delayed_Increasing,
        Until,
        Capacity,
        DelayedCapacity,
        Moran,
        Binomial,
        Between,
    )
    from .cython_stochastic import set_rng
except ModuleNotFoundError:
    print(
        "Couldn't find the compiled cython modules, fallback to python. This might be slow!"
    )
    from .python_definition import (
        CellType,
        Connection,
        Delayed,
        Delayed_Increasing,
        Until,
        Capacity,
        DelayedCapacity,
        Moran,
        Binomial,
        Between,
    )
    from .python_stochastic import set_rng
except ImportError as e:
    print(e)
    print(
        "Couldn't import cython modules, are GSL and LIBC installed? Falling back to python, this might be slow!"
    )
    from .python_definition import (
        CellType,
        Connection,
        Delayed,
        Delayed_Increasing,
        Until,
        Capacity,
        DelayedCapacity,
        Moran,
        Binomial,
        Between,
    )
    from .python_stochastic import set_rng
