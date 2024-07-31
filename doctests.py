import numpy as np
import doctest
import os
from src.karray import source_code


def repr_remove_dtype(x):
    string = np.array_repr(x)
    if ', dtype' in string:
        parts = string.split(', dtype')
        if 'int' in parts[1] or 'float' in parts[1]:
            return parts[0] + ','.join(parts[1].split(',')[1:]) + ')'
    return string


if __name__ == "__main__":
    os.makedirs(os.path.join(os.getcwd(), 'tests', 'data'), exist_ok=True)
    np.set_string_function(repr_remove_dtype, repr=True)
    failure_count, test_count = doctest.testmod(source_code, verbose=True, report=True)
    if failure_count == 0:
        print("All tests passed!")
    else:
        raise ValueError("Some tests failed!")
