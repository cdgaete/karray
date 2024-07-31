 

__version__ = "2024.3.7"
__author__ = 'Carlos Gaete-Morales'

__all__ = [
    'Array',
    'concat',
    'from_pandas',
    'from_polars',
    'from_feather_to_dict',
    'from_feather',
    'from_csv_to_dict',
    'from_csv',
    'ndarray_choice',
    'union_multi_coords',  # Require by symbolx
    'settings',
]

from .source_code import (
    Array,
    concat,
    from_pandas,
    from_polars,
    from_feather_to_dict,
    from_feather,
    from_csv_to_dict,
    from_csv,
    ndarray_choice,
    union_multi_coords,
    settings,
)
