import numpy as np
import doctest
import os
from src.karray import source_code


def format_array_no_dtype(arr):
    """
    Format array without dtype information, avoiding recursive calls.
    This is a direct string formatter that doesn't call back into numpy's repr system.
    """
    # For simple arrays, just convert to string and remove dtype if present
    if hasattr(arr, 'dtype'):
        # Handle basic scalar case
        if arr.ndim == 0:
            return str(arr.item())

        # Handle specific dtype cases
        if np.issubdtype(arr.dtype, np.integer):
            return str(arr.tolist())
        elif np.issubdtype(arr.dtype, np.floating):
            # Format floating point numbers with proper precision
            return str(arr.tolist())
        elif arr.dtype == np.bool_:
            return str(arr.tolist())
        elif arr.dtype.kind == 'U' or arr.dtype.kind == 'S':  # unicode or string
            return str(arr.tolist())

    # Fallback for other cases
    return str(arr)


if __name__ == "__main__":
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'data'), exist_ok=True)

    # The best approach with NumPy 2.0 is to NOT use a custom formatter for doctests
    # Instead, we should modify our doctest expectations or use NORMALIZE_WHITESPACE/ELLIPSIS
    # flags in doctest.testmod

    failure_count, test_count = doctest.testmod(
        source_code,
        verbose=True,
        report=True,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )

    if failure_count == 0:
        print("All tests passed!")
    else:
        raise ValueError(f"{failure_count} of {test_count} tests failed!")
