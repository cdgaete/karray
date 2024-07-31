import csv
import json
from importlib.util import find_spec
import numpy as np
from html import escape
from collections import Counter
from functools import reduce as functools_reduce
from typing import Any, Dict, Iterator, List, Tuple, Union, Callable
from .constants import css_style
try:
    import sparse as sp
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass
try:
    import pyarrow as pa
    from pyarrow import feather
except ImportError:
    pass
try:
    import cupy as cp
except ImportError:
    pass
try:
    import torch
except ImportError:
    pass


class Settings:
    def __init__(self) -> None:
        """
        Initialize the Settings object with default values.

        The Settings object holds various configuration options for the Array and Long objects.
        It includes settings for display, data types, and other behavior.

        """
        self.order = None
        self.rows_display = 16
        self.decimals_display = 2
        self.oneshot_display = False
        self.keep_zeros = False
        self.sort_coords = False
        self.fill_value = False
        self.data_type = "dense"  # 'dense' or 'sparse'
        self.df_with = "pandas"  # only for Array.df
        self.dense_dtype = None
        self.sparse_dtype = None
        self.long_dtype = None
        self.dataframe_as = "dense"  # 'dense' or 'sparse'
        self.gpu_backend = None  # 'cupy' or None
        self.gpu_backend_device = 0


settings = Settings()


def _isinstance_optional_pkgs(variable: Any, optional_packages_types: Union[str, Tuple[str]]) -> bool:
    """
    Check if a variable is an instance of any of the specified types from optional packages.

    This function is used to check if a variable is an instance of any of the types provided in the
    `optional_packages_types` argument. It handles cases where the optional packages may not be installed.

    Args:
        variable: The variable to check the type of.
        optional_packages_types: A string or tuple of strings representing the types to check against.
            The types should be specified as strings in the format "package.Type", e.g., "sp.COO".

    Returns:
        True if the variable is an instance of any of the specified types, False otherwise.

    Raises:
        AssertionError: If any of the specified types are not found in the `all_types_as_string` list.
            This indicates that the type should be added to the list or removed from the argument.

    Example:
        ```python
        >>> import sparse as sp
        >>> variable = sp.COO(data=[10, 20], coords=[[0, 1], [0, 1]], shape=(2, 2))
        >>> _isinstance_optional_pkgs(variable, 'sp.COO')
        True
        >>> variable = pd.DatetimeIndex(['2020-01-01', '2020-01-02'])
        >>> _isinstance_optional_pkgs(variable, ('sp.COO', 'pd.DatetimeIndex'))
        True

        ```

    """
    # This list must match the actual types shown below.
    all_types_as_string = ['sp.COO', 'pd.DatetimeIndex',
                           'pd.Categorical', 'pd.DataFrame', 'pl.DataFrame', 'pa.Table']
    if isinstance(optional_packages_types, str):
        optional_packages_types = (optional_packages_types,)
    not_found = []
    for type_str in optional_packages_types:
        for optional_package in ['sparse', 'pandas', 'polars', 'pyarrow']:
            if find_spec(name=optional_package) is not None:
                # List with types are provided here to avoid importing the package early
                if optional_package == 'sparse':
                    # if you add here more types, add them in the all_types_as_string
                    types_list = [sp.COO]

                elif optional_package == 'pandas':
                    # if you add here more types, add them in the all_types_as_string
                    types_list = [pd.DatetimeIndex,
                                  pd.Categorical, pd.DataFrame]

                elif optional_package == 'polars':
                    # if you add here more types, add them in the all_types_as_string
                    types_list = [pl.DataFrame]

                elif optional_package == 'pyarrow':
                    # if you add here more types, add them in the all_types_as_string
                    types_list = [pa.Table]
                for type_ in types_list:
                    if type_.__name__ in type_str:
                        if isinstance(variable, type_):
                            return True
                        else:
                            break
        if type_str not in all_types_as_string:
            not_found.append(type_str)
    assert len(
        not_found) == 0, f"Note to developers:The following optional types {not_found} should match types in the function _isinstance_optional_pkgs. Remove it from the argument or include them into the function."
    return False


class Long:
    def __init__(self, index: Dict[str, Union[np.ndarray, List[str], List[int], List[float], List[np.datetime64]]], value: Union[np.ndarray, List[float], List[int], List[bool], bool, int, float]) -> None:
        """
        Initialize a Long object.

        A Long object represents a multi-dimensional array in a sparse format, where the data is stored
        as a 1D value array and the corresponding coordinates are stored in the index dictionary.

        Args:
            index: A dictionary representing the index of the Long object. The keys are dimension names
                and the values are arrays or lists of coordinates.
            value: The value array of the Long object. It can be a NumPy array, list, or scalar value.

        Raises:
            AssertionError: If the index is not a dictionary, the index values are not arrays or lists,
                the index and value arrays have different lengths, the index keys are not strings,
                or 'value' is used as a dimension name.

        Example:
            ```python
            >>> index = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> value = [10, 20]
            >>> long_obj = Long(index, value)
            >>> long_obj
            Long(index={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, value=array([10, 20]))

            ```
        """
        value = _test_type_and_update_value(value)
        assert isinstance(index, dict), "Index must be a dictionary"
        dims = list(index)
        for dim in dims:
            assert isinstance(index[dim], (list, np.ndarray)) or _isinstance_optional_pkgs(index[dim], ('pd.DatetimeIndex', 'pd.Categorical')
                                                                                           ), "Index must be a dictionary with string keys and list, np.ndarray, pd.DatetimeIndex or pd.Categorical values"
            index[dim] = _test_type_and_update(index[dim])
        assert all([index[dim].size == value.size for dim in index]
                   ), "Index and value arrays must have the same length"
        assert all([isinstance(dim, str) for dim in index]), "Index must be a dictionary with string keys"
        assert 'value' not in index, "'value' can not be a dimension name as it is reserved"
        self.long_dtype = settings.long_dtype
        if self.long_dtype is not None:
            assert self.long_dtype in ["float16", "float32",
                                       "float64"], "settings.long_dtype must be 'float16', 'float32' or 'float64'"
            value = value.astype(self.long_dtype)
        self.value = value
        self.index = index
        self.dims = list(self.index)
        self.rows_display = settings.rows_display
        self.decimals_display = settings.decimals_display
        self.oneshot_display = settings.oneshot_display
        self.long_nbytes = _format_bytes(
            sum([self.index[dim].nbytes for dim in self.index] + [self.value.nbytes]))

    def __repr__(self) -> str:
        """
        Return a string representation of the Long object.

        Returns:
            A string representation of the Long object, including the index and value arrays.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> repr(long_obj)
            "Long(index={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, value=array([10, 20]))"

            ```
        """
        return f"Long(index={self.index}, value={repr(self.value)})"

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the Long object.

        The HTML representation includes a table displaying the Long object's size, dimensions, shape,
        capacity, and a preview of the data. The data preview shows a limited number of rows and rounds
        the values to a specified number of decimal places.

        Returns:
            An HTML string representation of the Long object.
        """
        dims = self.dims
        items = self.value.size
        if items > self.rows_display:
            short = False
            if self.oneshot_display:
                rows = self.rows_display
            else:
                rows = int(self.rows_display/2)
        else:
            short = True
            rows = items
        columns = dims + ['value']
        html = [f"{css_style}"]
        html += ['<h3>[Long]</h3>',
                 '<table>',
                 f'<tr><th>Long object size</th><td>{self.long_nbytes}</td></tr>',
                 "<!-- DATA_TYPE -->",
                 "<!-- DATA -->",
                 f'<tr><th>Dimensions</th><td>{dims}</td></tr>',
                 '<!-- SHAPE -->',
                 '<!-- capacity -->',
                 f'<tr><th>Rows</th><td>{items}</td></tr>',
                 '</table>']
        html += ["<!-- COORDS -->"]
        html += ["<details>"]
        html += ['<table><summary><div class="tooltip"> Show data <small>[default: 16 rows, 2 decimals]</small>']
        html += ['<!-- A --><span class="tooltiptext tooltip-top">To change default values:<br> obj.rows_display = Int val<br>obj.decimals_display = Int val<br>obj.oneshot_display = False<!-- Z -->']
        html += ['</span></div></summary><tr><th>']
        html += [f"<th>{j}" for j in columns]
        for i in range(rows):
            html.append(f"<tr><th><b>{i}</b>")
            for j, v in self.items():
                val = v[i]
                html.append("<td>")
                html.append(escape(f"{val:.{self.decimals_display}f}" if issubclass(
                    v.dtype.type, (np.float16, np.float32, np.float64)) else f"{val}"))
        if not self.oneshot_display:
            if not short:
                html.append("<tr><th>")
                for _ in range(len(dims)+1):
                    html.append("<td>...")
                for i in range(items-rows, items, 1):
                    html.append(f"<tr><th><b>{i}</b>")
                    for j, v in self.items():
                        val = v[i]
                        html.append("<td>")
                        html.append(escape(f"{val:.{self.decimals_display}f}" if issubclass(
                            v.dtype.type, (np.float16, np.float32, np.float64)) else f"{val}"))
        html.append("</table></details>")
        return "".join(html)

    @property
    def size(self) -> int:
        """
        Get the number of elements in the Long object.

        Returns:
            The size of the Long object, which is the number of elements in the value array.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj.size
            2

            ```
        """
        return self.value.size

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the Long object.

        Returns:
            The number of dimensions of the Long object, which is the number of keys in the index dictionary.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj.ndim
            2

            ```
        """
        return len(self.index)

    def insert(self, **kwargs: Dict[str, Union['np.dtype[Any]', type, str, int, Dict[str, Union[Dict[Any, Any], List[Union[np.ndarray, np.ndarray]]]]]]) -> 'Long':
        """
        Insert new dimensions into the Long object.

        Args:
            **kwargs: Keyword arguments specifying the new dimensions and their values. The keys represent
                the names of the new dimensions, and the values can be of the following types:
                - np.dtype or type: Specifies the data type of the new dimension. Only valid for empty arrays.
                - str, int, or float: Specifies a single value for the new dimension.
                - dict: Specifies a mapping between an existing dimension and the new dimension. The keys
                  of the dict represent the existing dimension, and the values can be either a dict mapping
                  old values to new values, or a list of two lists representing the old and new values.

        Returns:
            A new Long object with the inserted dimensions.

        Raises:
            AssertionError: If the new dimension names already exist in the existing dimensions,
                the new dimensions items are not of the supported types, or the mapping between dimensions
                is invalid.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> new_long_obj = long_obj.insert(dim3=1)
            >>> new_long_obj.dims
            ['dim3', 'dim1', 'dim2']

            ```
        """
        assert all([dim not in self.dims for dim in kwargs]
                   ), "new dimension names must not exist in the existing dimensions"
        assert all([isinstance(kwargs[dim], (str, int, float, dict, np.dtype, type))
                   for dim in kwargs]), "new dimensions items must be a str, int, float, dict, np.dtype or type"
        # When dict is passed, it contains a map of old_dim -> new_dim as nested dict or a list of two lists.
        for dim in kwargs:
            if isinstance(kwargs[dim], dict):
                value = kwargs[dim]
                assert len(value) == 1
                existing_dim = next(iter(value))
                assert isinstance(existing_dim, (str, tuple))
                if isinstance(existing_dim, tuple):
                    assert all([dim in self.dims for dim in existing_dim])
                else:
                    assert existing_dim in self.dims
                assert isinstance(value[existing_dim], (dict, list))
                if isinstance(existing_dim, str):
                    if isinstance(value[existing_dim], dict):
                        old_dim_items = list(value[existing_dim])
                        old_dim_items_set = set(old_dim_items)
                    elif isinstance(value[existing_dim], list):
                        kwargs[dim][existing_dim][0] = _test_type_and_update(
                            value[existing_dim][0])
                        kwargs[dim][existing_dim][1] = _test_type_and_update(
                            kwargs[dim][existing_dim][1])
                        old_dim_items = kwargs[dim][existing_dim][0]
                        old_dim_items_set = set(old_dim_items)
                    assert set(np.unique(self.index[existing_dim])).issubset(
                        old_dim_items_set)
                    assert len(old_dim_items) == len(old_dim_items_set)
                elif isinstance(existing_dim, tuple):
                    if isinstance(value[existing_dim], dict):
                        raise NotImplementedError("TODO")
                    elif isinstance(value[existing_dim], list):
                        kwargs[dim][existing_dim][1] = _test_type_and_update(
                            kwargs[dim][existing_dim][1])
        index = {}
        for new_dim in kwargs:
            value = kwargs[new_dim]
            # when dtype or type object is passed, the addition of new dimenssion is only possible to a empty array.
            if isinstance(value, (np.dtype, type)):
                assert self.value.size == 0, "new dimensions type setting cannot be performed to non-empty arrays"
                idxarray = np.empty(self.size, dtype=value)
            elif isinstance(value, str):
                idxarray = np.empty(self.size, dtype=np.object_)
                idxarray[:] = value
            elif isinstance(value, int):
                idxarray = np.empty(self.size, dtype=np.int32)
                idxarray[:] = value
            elif isinstance(value, float):
                idxarray = np.empty(self.size, dtype=np.float32)
                idxarray[:] = value
            elif isinstance(value, dict):
                existing_dim = next(iter(value))
                if isinstance(existing_dim, str):
                    if isinstance(value[existing_dim], dict):
                        mapping_dict = value[existing_dim]
                        existing_dim_items = self.index[existing_dim]
                        k = np.array(list(mapping_dict))
                        v = np.array(list(mapping_dict.values()))
                    elif isinstance(value[existing_dim], list):
                        assert isinstance(value[existing_dim][0], np.ndarray)
                        assert isinstance(value[existing_dim][1], np.ndarray)
                        k = value[existing_dim][0]
                        v = value[existing_dim][1]
                        existing_dim_items = self.index[existing_dim]
                    else:
                        raise Exception(
                            f"type {type(value[existing_dim])} not implemented.")
                    idxarray = np.array(v)[np.argsort(k)[np.searchsorted(
                        k, existing_dim_items, sorter=np.argsort(k))]]
                elif isinstance(existing_dim, tuple):
                    if isinstance(value[existing_dim], dict):
                        raise NotImplementedError("TODO")
                    elif isinstance(value[existing_dim], list):
                        assert isinstance(value[existing_dim][1], np.ndarray)
                        coords = value[existing_dim][0]
                        new_dim_elements = value[existing_dim][1]
                        collect_index = []
                        for dim in existing_dim:
                            a = np.argsort(coords[dim])[np.searchsorted(
                                coords[dim], self.index[dim], sorter=np.argsort(coords[dim]))]
                            collect_index.append(a)
                        index_index = np.vstack(collect_index)
                        shape = [coords[dim].size for dim in coords]
                        indexes = np.ravel_multi_index(index_index, shape)
                        idxarray = new_dim_elements[indexes]
                    else:
                        raise Exception(
                            f"type {type(value[existing_dim])} not implemented.")
            index[new_dim] = idxarray
        for dim in self.index:
            index[dim] = self.index[dim]
        return Long(index=index, value=self.value)

    def rename(self, **kwargs: str) -> 'Long':
        """
        Rename dimensions of the Long object.

        Args:
            **kwargs: Keyword arguments specifying the old dimension names and their new names. The keys
                represent the old dimension names, and the values represent the new dimension names.

        Returns:
            A new Long object with the renamed dimensions.

        Raises:
            AssertionError: If the old dimension names do not exist in the current dimensions or the new
                dimension names already exist in the current dimensions.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> new_long_obj = long_obj.rename(dim1='new_dim1')
            >>> new_long_obj.dims
            ['new_dim1', 'dim2']

            ```
        """
        assert all([odim in self.dims for odim in kwargs])
        assert all([ndim not in self.dims for ndim in kwargs.values()])
        index = {}
        for dim in self.dims:
            if dim in kwargs:
                index[kwargs[dim]] = self.index[dim]
            else:
                index[dim] = self.index[dim]
        return Long(index=index, value=self.value)

    def drop(self, dims: Union[str, List[str]]) -> 'Long':
        """
        Drop specified dimensions from the Long object.

        Args:
            dims: A single dimension or a list of dimensions to drop.

        Returns:
            A new Long object with the specified dimensions dropped.

        Raises:
            AssertionError: If the specified dimensions do not exist in the current dimensions or dropping
                the dimensions results in non-unique index items per row.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2], 'dim3': ['a', 'a']}, value=[10, 20])
            >>> new_long_obj = long_obj.drop('dim3')
            >>> new_long_obj.dims
            ['dim1', 'dim2']

            ```
        """
        assert isinstance(dims, (str, list))
        index = {}
        if isinstance(dims, str):
            assert dims in self.dims
            dims = [dims]
        elif isinstance(dims, list):
            assert all([dim in self.dims for dim in dims])
        for dim in self.dims:
            if dim not in dims:
                index[dim] = self.index[dim]
        # TODO: this can be slow. Check if it can be optimized.
        # Compare with _check_duplicate_indexes
        item_tuples = list(zip(*index.values()))
        if len(set(item_tuples)) == len(item_tuples):
            flag = True
        else:
            flag = False
            counts = Counter(item_tuples)
            most_common = counts.most_common(1)[0][0]
            first = item_tuples.index(most_common, 0)
            second = item_tuples.index(most_common, first+1)
            display_str = f"e.g.:\n  {tuple(index)} value\n{first} {item_tuples[first]} {self.value[first]}\n{second} {item_tuples[second]} {self.value[second]}"
        assert flag, f"Index items per row must be unique. By removing {dims} leads the existence of repeated indexes \n{display_str}\nIntead, you can use obj.reduce('{dims[0]}')\nWith an aggfunc: sum() by default"
        return Long(index=index, value=self.value)

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Iterate over the dimensions and value arrays of the Long object.

        Yields:
            A tuple of dimension name and its corresponding value array.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> for item in long_obj.items():
            ...     print(item)
            ('dim1', array(['a', 'b'], dtype=object))
            ('dim2', array([1, 2]))
            ('value', array([10, 20]))

            ```
        """
        dc = dict(**self.index)
        dc.update(dict(value=self.value))
        for k, v in dc.items():
            yield (k, v)

    def __getitem__(self, item: Union[str, int, List[Any], np.ndarray, slice, Tuple[str, Union[List[Any], np.ndarray, slice]]]) -> 'Long':
        """
        Get a subset of the Long object based on the provided index.

        Args:
            item: An index specifier to subset the Long object. It can be one of the following types:
                - str: Selects a specific dimension.
                - int: Selects a specific row by index.
                - list or np.ndarray: Selects specific rows by index.
                - slice: Selects a range of rows.
                - tuple: Selects specific elements based on a dimension and a condition. The first element
                  of the tuple is the dimension name, and the second element is the condition (list, np.ndarray, or slice).

        Returns:
            A new Long object representing the subset based on the provided index.

        Raises:
            AssertionError: If the provided index is invalid or the specified dimension does not exist.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> subset_long_obj = long_obj['dim1', ['a']]
            >>> subset_long_obj.index
            {'dim1': array(['a'], dtype=object), 'dim2': array([1])}
            >>> subset_long_obj.value
            array([10])

            ```
        """
        assert isinstance(item, (str, int, list, np.ndarray, slice, tuple))
        if isinstance(item, int):
            return Long(index={dim: self.index[dim][item] for dim in self.dims}, value=self.value[item])
        elif isinstance(item, list):
            item = np.array(item, dtype=np.int32)
            return Long(index={dim: self.index[dim][item] for dim in self.dims}, value=self.value[item])
        elif isinstance(item, np.ndarray):
            assert issubclass(item.dtype.type, (np.int16, np.int32, np.int64)) or issubclass(item.dtype.type, np.bool_)
            return Long(index={dim: self.index[dim][item] for dim in self.dims}, value=self.value[item])
        elif isinstance(item, slice):
            return Long(index={dim: self.index[dim][item] for dim in self.dims}, value=self.value[item])
        elif isinstance(item, str):
            assert item in self.dims
            return self.index[item]
        elif isinstance(item, tuple):
            assert len(item) == 2
            if isinstance(item[0], str):
                dim = item[0]
                condition = item[1]
                assert dim in self.dims
                assert isinstance(condition, (list, np.ndarray, slice))
                index_items_on_dim = self.index[dim]
                if isinstance(condition, (list, np.ndarray)):
                    mask = np.isin(index_items_on_dim, condition)
                    return Long(index={dim_: self.index[dim_][mask] for dim_ in self.dims}, value=self.value[mask])
                elif isinstance(condition, slice):
                    assert issubclass(index_items_on_dim.dtype.type, (np.int16, np.int32, np.int64))
                    start = condition.start or int(np.min(index_items_on_dim))
                    step = condition.step or 1
                    stop = condition.stop or int(
                        np.max(index_items_on_dim) + step)
                    arange_condition = np.arange(start, stop, step)
                    mask = np.isin(index_items_on_dim, arange_condition)
                    return Long(index={dim_: self.index[dim_][mask] for dim_ in self.dims}, value=self.value[mask])
            elif isinstance(item[0], list):
                reorder = item[0]
                assert set(self.dims) == set(reorder)
                assert isinstance(item[1], slice)
                condition = item[1]
                start = condition.start or 0
                stop = condition.stop or self.value.size
                step = condition.step or 1
                arange_condition = np.arange(start, stop, step)
                return Long(index={dim_: self.index[dim_][arange_condition] for dim_ in reorder}, value=self.value[arange_condition])

    def __eq__(self, other: Union['Long', float, int, np.generic]) -> Union[bool, np.ndarray]:
        """
        Check equality between the Long object and another object.

        Args:
            other: Another Long object or a scalar value.

        Returns:
            True if the objects are equal, False otherwise. If compared with a scalar value, returns a boolean array.

        Raises:
            Exception: If the comparison is not supported for the given type.

        Example:
            ```python
            >>> long_obj1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj1 == long_obj2
            True
            >>> long_obj1 == 10
            array([ True, False])

            ```
        """
        if isinstance(other, Long):
            dims_equal = tuple(self.dims) == tuple(other.dims)
            if not dims_equal:
                return False
            value_equal = np.array_equal(self.value, other.value)
            if not value_equal:
                return False
            return all(np.array_equal(self.index[dim], other.index[dim]) for dim in self.dims)
        else:
            if np.isnan(other):
                return np.isnan(self.value)
            elif np.isinf(other):
                return np.isinf(self.value)
            elif isinstance(other, (int, float)):
                return self.value == other
            elif isinstance(other, np.generic):
                raise Exception("np.ndarray not supported yet")
            else:
                raise Exception(f"{type(other)} not supported yet")

    def __ne__(self, other: Union['Long', float, int, np.generic]) -> Union[bool, np.ndarray]:
        """
        Check inequality between the Long object and another object.

        Args:
            other: Another Long object or a scalar value.

        Returns:
            True if the objects are not equal, False otherwise. If compared with a scalar value, returns a boolean array.

        Raises:
            Exception: If the comparison is not supported for the given type.

        Example:
            ```python
            >>> long_obj1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 30])
            >>> long_obj1 != long_obj2
            True
            >>> long_obj1 != 10
            array([False,  True])

            ```
        """
        if isinstance(other, Long):
            dims_equal = tuple(self.dims) == tuple(other.dims)
            if not dims_equal:
                return True
            value_equal = np.array_equal(self.value, other.value)
            if not value_equal:
                return True
            return not all(np.array_equal(self.index[dim], other.index[dim]) for dim in self.dims)
        else:
            if np.isnan(other):
                return ~np.isnan(self.value)
            elif np.isinf(other):
                return ~np.isinf(self.value)
            elif isinstance(other, (int, float)):
                return self.value != other
            elif isinstance(other, np.generic):
                raise Exception("np.ndarray not supported yet")
            else:
                raise Exception(f"{type(other)} not supported yet")

    def __lt__(self, other: Union[float, int, np.ndarray, 'Long']) -> np.ndarray:
        """
        Check if the Long object is less than another object element-wise.

        Args:
            other: A scalar value, numpy array, or another Long object.

        Returns:
            A boolean array indicating whether each element of the Long object is less than the corresponding element of the other object.

        Raises:
            Exception: If the operation is not supported for the given type.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj < 15
            array([ True, False])
            >>> long_obj < np.array([15, 15])
            array([ True, False])

            ```
        """
        if isinstance(other, (int, float)):
            return np.less(self.value, other)
        elif isinstance(other, np.ndarray):
            return np.less(self.value, other)
        elif isinstance(other, Long):
            return np.less(self.value, other.value)
        else:
            raise Exception(f"Operation not supported on {type(other)}")

    def __le__(self, other: Union[float, int, np.ndarray, 'Long']) -> np.ndarray:
        """
        Check if the Long object is less than or equal to another object element-wise.

        Args:
            other: A scalar value, numpy array, or another Long object.

        Returns:
            A boolean array indicating whether each element of the Long object is less than or equal to the corresponding element of the other object.

        Raises:
            Exception: If the operation is not supported for the given type.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj <= 15
            array([ True, False])
            >>> long_obj <= np.array([10, 20])
            array([ True,  True])

            ```
        """
        if isinstance(other, (int, float)):
            return np.less_equal(self.value, other)
        elif isinstance(other, np.ndarray):
            return np.less_equal(self.value, other)
        elif isinstance(other, Long):
            return np.less_equal(self.value, other.value)
        else:
            raise Exception(f"Operation not supported on {type(other)}")

    def __gt__(self, other: Union[float, int, np.ndarray, 'Long']) -> np.ndarray:
        """
        Check if the Long object is greater than another object element-wise.

        Args:
            other: A scalar value, numpy array, or another Long object.

        Returns:
            A boolean array indicating whether each element of the Long object is greater than the corresponding element of the other object.

        Raises:
            Exception: If the operation is not supported for the given type.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj > 15
            array([False,  True])
            >>> long_obj > np.array([15, 15])
            array([False,  True])

            ```
        """
        if isinstance(other, (int, float)):
            return np.greater(self.value, other)
        elif isinstance(other, np.ndarray):
            return np.greater(self.value, other)
        elif isinstance(other, Long):
            return np.greater(self.value, other.value)
        else:
            raise Exception(f"Operation not supported on {type(other)}")

    def __ge__(self, other: Union[float, int, np.ndarray, 'Long']) -> np.ndarray:
        """
        Check if the Long object is greater than or equal to another object element-wise.

        Args:
            other: A scalar value, numpy array, or another Long object.

        Returns:
            A boolean array indicating whether each element of the Long object is greater than or equal to the corresponding element of the other object.

        Raises:
            Exception: If the operation is not supported for the given type.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long_obj >= 15
            array([False,  True])
            >>> long_obj >= np.array([10, 20])
            array([ True,  True])

            ```
        """
        if isinstance(other, (int, float)):
            return np.greater_equal(self.value, other)
        elif isinstance(other, np.ndarray):
            return np.greater_equal(self.value, other)
        elif isinstance(other, Long):
            return np.greater_equal(self.value, other.value)
        else:
            raise Exception(f"Operation not supported on {type(other)}")

    def to_pandas(self) -> 'pd.DataFrame':
        """
        Convert the Long object to a pandas DataFrame.

        Returns:
            A pandas DataFrame representing the Long object.

        Example:
            ```python
            >>> long_obj = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> df = long_obj.to_pandas()
            >>> df
              dim1  dim2  value
            0    a     1     10
            1    b     2     20

            ```
        """
        data = {dim: self.index[dim] for dim in self.dims}
        data['value'] = self.value
        return pd.DataFrame(data=data)


class Array:
    def __init__(self, data: Union[Tuple[dict, Union[np.ndarray, List[float], List[int], List[bool]]], Long, np.ndarray, 'sp.COO', None] = None, coords: Union[Dict[str, Union[np.ndarray, List[str], List[int], List[float], List[np.datetime64]]], None] = None) -> None:
        """
        Initialize an Array object.

        Args:
            data: The data for the Array object. It can be a tuple of index and value, a Long object, a dense numpy array, a sparse COO array, or None.
            coords: A dictionary representing the coordinates of the Array object.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=long, coords=coords)
            >>> arr
            Array(data=array([[10,  0],
                   [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        self.__dict__["_repo"] = {}
        self.long = None
        self.coords = None
        self.dense = None
        self.sparse = None
        self.data_type = settings.data_type
        self.keep_zeros = settings.keep_zeros
        self.sort_coords = settings.sort_coords
        self.fill_value = settings.fill_value
        self.df_with = settings.df_with
        self.order = settings.order
        self.dense_dtype = settings.dense_dtype
        self.sparse_dtype = settings.sparse_dtype
        self.dataframe_as = settings.dataframe_as
        self.gpu_backend = settings.gpu_backend
        self.gpu_backend_device = settings.gpu_backend_device
        self._attr_constructor(**self._check_input(data, coords))
        return None

    def _check_input(self, data: Union[Tuple[dict, Union[np.ndarray, List[float], List[int], List[bool]]], Long, np.ndarray, 'sp.COO', None], coords: Union[Dict[str, Union[np.ndarray, List[str], List[int], List[float], List[np.datetime64]]], None]) -> Union[Tuple[dict, Union[np.ndarray, List[float], List[int], List[bool]]], Long, np.ndarray, 'sp.COO', None]:
        """
        Check the input for the Array object.

        Args:
            data: The data for the Array object. It can be a tuple of index and value, a Long object, a dense numpy array, a sparse COO array, or None.
            coords: A dictionary representing the coordinates of the Array object.

        Returns:
            A tuple of index and value, a Long object, a dense numpy array, a sparse COO array, or None.

        Raises:
            AssertionError: If the input data or coordinates are invalid.
        """
        assert data is None or isinstance(data, (tuple, Long, np.ndarray)) or _isinstance_optional_pkgs(
            data, 'sp.COO'), f"Invalid type for 'data': {type(data)}"
        assert coords is None or isinstance(
            coords, dict), f"Invalid type for 'coords': {type(coords)}"
        if isinstance(data, Long):
            long: Union[Long, None] = data
            index: Union[dict, None] = None
            value: Union[np.ndarray, None] = None
            dense: Union[np.ndarray, None] = None
            sparse: Union['sp.COO', None] = None
        elif isinstance(data, tuple):
            long: Union[Long, None] = None
            index: Union[dict, None] = data[0]
            value: Union[np.ndarray, None] = data[1]
            dense: Union[np.ndarray, None] = None
            sparse: Union['sp.COO', None] = None
        elif isinstance(data, np.ndarray):
            long: Union[Long, None] = None
            index: Union[dict, None] = None
            value: Union[np.ndarray, None] = None
            dense: Union[np.ndarray, None] = data
            sparse: Union['sp.COO', None] = None
            assert coords is not None
        elif _isinstance_optional_pkgs(data, 'sp.COO'):
            long: Union[Long, None] = None
            index: Union[dict, None] = None
            value: Union[np.ndarray, None] = None
            dense: Union[np.ndarray, None] = None
            sparse: Union['sp.COO', None] = data
        else:
            long: Union[Long, None] = None
            index: Union[dict, None] = None
            value: Union[np.ndarray, None] = None
            dense: Union[np.ndarray, None] = None
            sparse: Union['sp.COO', None] = None
            assert coords is not None
        if isinstance(data, tuple):
            assert isinstance(index, dict), "Index must be a dictionary. "
            for dim in index:
                assert isinstance(
                    dim, str), "Index must be a dictionary with string keys"
                assert isinstance(index[dim], (list, np.ndarray)) or _isinstance_optional_pkgs(index[dim], ('pd.DatetimeIndex', 'pd.Categorical')
                                                                                               ), "Index must be a dictionary with string keys and list, np.ndarray, pd.DatetimeIndex or pd.Categorical values"
                index[dim] = _test_type_and_update(index[dim])
            value = _test_type_and_update_value(value)
        if coords is not None:
            assert isinstance(coords, dict), "coords must be a dictionary"
            assert all([isinstance(coords[dim], (np.ndarray, list)) or _isinstance_optional_pkgs(coords[dim], ('pd.DatetimeIndex', 'pd.Categorical'))
                       for dim in coords]), "coords must contains list, np.ndarray values"
            cdims = list(coords)
            for dim in cdims:
                assert isinstance(
                    dim, str), "coords must be a dictionary with string keys"
                coords[dim] = _test_type_and_update(coords[dim])
                assert coords[dim].ndim == 1
                assert coords[dim].size == np.unique(
                    coords[dim]).size, f"coords elements of dim '{dim}' must be unique. {coords[dim].size=}, {np.unique(coords[dim]).size=}"
            if long is not None:
                assert set(long.dims) == set(list(coords))
            elif index is not None:
                assert set(list(index)) == set(list(coords))
            elif dense is not None:
                assert dense.ndim == len(coords)
                assert dense.shape == tuple(self._shape(coords))
                assert dense.size == self._capacity(coords)
            elif sparse is not None:
                assert sparse.ndim == len(coords)
                assert sparse.shape == tuple(self._shape(coords))
                assert sparse.nnz <= self._capacity(coords)

        return dict(sparse=sparse, dense=dense, long=long, index=index, value=value, coords=coords)

    def _attr_constructor(self, sparse: 'sp.COO', dense: np.ndarray, long: Long, index: Dict[str, np.ndarray], value: np.ndarray, coords: Dict[str, np.ndarray]) -> None:
        """
        Set the attributes of the Array object based on the provided data and coordinates.

        Args:
            sparse: The sparse COO array.
            dense: The dense array.
            long: The Long object.
            index: The index of the array.
            value: The value of the array.
            coords: The coordinates of the array.

        Returns:
            None
        """
        if long is not None:
            if coords is not None:
                if len(coords) == 0:
                    assert long.ndim == 0
                    self.coords = self._reorder_coords(
                        coords, self.order, self.sort_coords)
                    self.long = self._reorder_long(
                        long, list(self.coords), self.keep_zeros)
                else:
                    self.coords = self._reorder_coords(
                        coords, self.order, self.sort_coords)
                    self.long = self._reorder_long(
                        long, list(self.coords), self.keep_zeros)
            else:
                coords = {dim: np.sort(
                    np.unique(long.index[dim])) for dim in long.dims}
                self.coords = self._reorder_coords(
                    coords, self.order, self.sort_coords)
                self.long = self._reorder_long(
                    long, list(self.coords), self.keep_zeros)
        elif index is not None:
            if value is None:
                raise Exception(
                    "If 'index' is not None, then 'value' must be provided. Currently 'value' is None")
            else:
                if coords is not None:
                    self.coords = self._reorder_coords(
                        coords, self.order, self.sort_coords)
                    assert set(self.coords) == set(index)
                    index = {dim: index[dim] for dim in self.coords}
                    long = Long(index=index, value=value)
                    self.long = self._reorder_long(
                        long, list(self.coords), self.keep_zeros)
                else:
                    coords = {dim: np.sort(
                        np.unique(index[dim])) for dim in index}
                    self.coords = self._reorder_coords(
                        coords, self.order, self.sort_coords)
                    index = {dim: index[dim] for dim in self.coords}
                    long = Long(index=index, value=value)
                    self.long = self._reorder_long(
                        long, list(self.coords), self.keep_zeros)
        elif dense is not None:
            assert coords is not None
            if tuple(self._order_with_preference(list(coords), self.order)) == tuple(list(coords)):
                if self.sort_coords:
                    self.coords = self._reorder_coords(
                        coords, self.order, self.sort_coords)
                    long = self._dense_to_long(dense, coords)
                    self.dense = self._dense(long, self.coords)
                    self.long = self._reorder_long(
                        long, list(self.coords), self.keep_zeros)
                else:
                    self.coords = coords
                    if issubclass(dense.dtype.type, (np.int64, np.int32, np.int16)):
                        dense = dense.astype(float)
                    self.long = self._dense_to_long(dense, coords)
                    self.dense = dense
            else:
                self.coords = self._reorder_coords(
                    coords, self.order, self.sort_coords)
                long = self._dense_to_long(dense, coords)
                self.dense = self._dense(long, self.coords)
                self.long = self._reorder_long(
                    long, list(self.coords), self.keep_zeros)
        elif sparse is not None:
            assert coords is not None
            if tuple(self._order_with_preference(list(coords), self.order)) == tuple(list(coords)):
                if self.sort_coords:
                    self.coords = self._reorder_coords(
                        coords, self.order, self.sort_coords)
                    long = self._sparse_to_long(sparse, coords)
                    self.sparse = self._sparse(long, self.coords)
                    self.long = self._reorder_long(
                        long, list(self.coords), self.keep_zeros)
                else:
                    self.coords = coords
                    self.sparse = sparse
                    self.long = self._sparse_to_long(sparse, coords)
            else:
                self.coords = self._reorder_coords(
                    coords, self.order, self.sort_coords)
                long = self._sparse_to_long(sparse, coords)
                self.sparse = self._sparse(long, self.coords)
                self.long = self._reorder_long(
                    long, list(self.coords), self.keep_zeros)
        else:
            if value is None:
                assert value is None and index is None and coords is not None
                self.coords = self._reorder_coords(
                    coords, self.order, self.sort_coords)
                dtypes = {dim: self.coords[dim].dtype.type for dim in coords}
                if len(coords) == 0:
                    long = Long(index={}, value=np.array([], dtype=float))
                    self.long = long
                else:
                    long = Long(index={dim: np.array([], dtype=dtypes[dim])
                                for dim in self.coords}, value=np.array([], dtype=float))
                    self.long = long
            else:
                raise Exception(
                    "If 'value' is not None, then 'index' must be provided. Currently 'index' is None")
        return None

    def __repr__(self) -> str:
        """
        Return a string representation of the Array object.

        Returns:
            A string representation of the Array object.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'a'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long)
            >>> repr(arr)
            "Array(data=array([[10, 20]]), coords={'dim1': array(['a'], dtype=object), 'dim2': array([1, 2])})"

            ```
        """
        return f"Array(data={repr(self.data)}, coords={self.coords})"

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the Array object.

        Returns:
            An HTML string representation of the Array object.
        """
        html = [
            '<details><table><summary><div class="tooltip"> Show unique coords</div></summary>']
        html.append("<tr><th>Dimension<th>Length<th>Type<th>Items")
        for dim in self.coords:
            html.append(f"<tr><th><b>{dim}</b><td>")
            html.append(escape(f"{len(self.coords[dim])}"))
            html.append("<td>")
            html.append(escape(f"{self.coords[dim].dtype}"))
            html.append("<td>")
            html.append('<details>')
            html.append(
                '<summary><div class="tooltip">show details</div></summary>')
            html.append(escape(f"{self.coords[dim]}"))
            html.append("</details>")
        html.append("</table></details>")
        data_type = f"<tr><th>Data object type</th><td>{self.data_type}</td></tr>"
        data_size = f"<tr><th>Data object size</th><td>{_format_bytes(self.data.nbytes)}</td></tr>"
        script = ''.join(html)
        shape = f"<tr><th>Shape</th><td>{self.shape}</td></tr>"
        capacity = f"<tr><th>Capacity</th><td>{self.capacity}</td></tr>"
        return self.long._repr_html_().replace('[Long]', '[k]array') \
                                      .replace('<!-- DATA_TYPE -->', data_type) \
                                      .replace('<!-- DATA -->', data_size) \
                                      .replace('<!-- COORDS -->', script) \
                                      .replace('<!-- SHAPE -->', shape) \
                                      .replace('<!-- capacity -->', capacity) \
                                      .replace('<!-- A -->', '<!-- ') \
                                      .replace('<!-- Z -->', ' -->')

    def _reorder_coords(self, coords: Dict[str, np.ndarray], order_preference: List[str], sort_coords: bool) -> Dict[str, np.ndarray]:
        """
        Reorder the coordinates based on the specified order preference and optionally sort them.

        Args:
            coords: A dictionary of coordinates.
            order_preference: The preferred order of dimensions.
            sort_coords: Whether to sort the coordinates.

        Returns:
            A new dictionary of coordinates with the specified order and sorting.

        Example:
            ```python
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> order_preference = ['dim2', 'dim1']
            >>> sort_coords = True
            >>> arr = Array(coords=coords)
            >>> arr._reorder_coords(coords, order_preference, sort_coords)
            {'dim2': array([1, 2]), 'dim1': array(['a', 'b'], dtype=object)}

            ```
        """
        order = self._order_with_preference(list(coords), order_preference)
        if sort_coords:
            coords_ = {dim: np.sort(coords[dim]) for dim in order}
        else:
            coords_ = {dim: coords[dim] for dim in order}
        return coords_

    def _reorder_long(self, long: Long, order: List[str], keep_zeros: bool) -> Long:
        """
        Reorder the Long object based on the specified order and optionally keep zero values.

        Args:
            long: The Long object to reorder.
            order: The desired order of dimensions.
            keep_zeros: Whether to keep zero values.

        Returns:
            The reordered Long object.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 0])
            >>> order = ['dim2', 'dim1']
            >>> keep_zeros = False
            >>> arr = Array(data=long)
            >>> arr._reorder_long(long, order, keep_zeros)
            Long(index={'dim2': array([1]), 'dim1': array(['a'], dtype=object)}, value=array([10]))

            ```
        """
        long = long[order, :]
        return long if keep_zeros else long[long != 0.0]

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute of the Array object.

        Args:
            name: The name of the attribute.
            value: The value to set for the attribute.

        Returns:
            None

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> arr = Array(data=long, coords=coords)
            >>> arr.dense_dtype = 'float32'

            ```
        """
        if name == "dense":
            if value is not None and self.dense_dtype is not None:
                if issubclass(value.dtype.type, (np.float16, np.float32, np.float64)):
                    assert self.dense_dtype in [
                        "float16", "float32", "float64"]
                    value = value.astype(self.dense_dtype)
        elif name == "sparse":
            if value is not None and self.sparse_dtype is not None:
                if issubclass(value.dtype.type, (np.float16, np.float32, np.float64)):
                    assert self.sparse_dtype in [
                        "float16", "float32", "float64"]
                    value = value.astype(self.sparse_dtype)
        elif name == "dense_dtype":
            if value is not None:
                assert value in ["float16", "float32", "float64"]
                self._repo['dense'] = self._repo['dense'].astype(
                    value) if self._repo['dense'] is not None else None
        elif name == "sparse_dtype":
            if value is not None:
                assert value in ["float16", "float32", "float64"]
                self._repo['sparse'] = self._repo['sparse'].astype(
                    value) if self._repo['sparse'] is not None else None
        self._repo[name] = value

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute of the Array object.

        Args:
            name: The name of the attribute.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the attribute is not found.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> arr = Array(data=long, coords=coords)
            >>> arr.long
            Long(index={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, value=array([10, 20]))

            ```
        """
        if name.startswith('_'):
            raise AttributeError(name)
        elif name == 'long':
            if name in self._repo:
                if self._repo[name] is None:
                    assert self.data is not None
                    if self.data_type == 'sparse':
                        self._repo[name] = self._sparse_to_long(
                            self.data, self.coords)
                    elif self.data_type == 'dense':
                        self._repo[name] = self._dense_to_long(
                            self.data, self.coords)
                    else:
                        raise ValueError(
                            f"data_type must be 'sparse' or 'dense', not {self.data_type}")
                    return self._repo[name]
                else:
                    return self._repo[name]
            else:
                assert self.data is not None
                if self.data_type == 'sparse':
                    self._repo[name] = self._sparse_to_long(
                        self.data, self.coords)
                elif self.data_type == 'dense':
                    self._repo[name] = self._dense_to_long(
                        self.data, self.coords)
                else:
                    raise ValueError(
                        f"data_type must be 'sparse' or 'dense', not {self.data_type}")
                return self._repo[name]
        elif name == 'dense':
            if name in self._repo:
                if self._repo[name] is None:
                    assert self.long is not None
                    self._repo[name] = self._dense(self.long, self.coords)
                    return self._repo[name]
                else:
                    return self._repo[name]
            else:
                assert self.long is not None
                self._repo[name] = self._dense(self.long, self.coords)
                return self._repo[name]
        elif name == 'sparse':
            if name in self._repo:
                if self._repo[name] is None:
                    assert self.long is not None
                    self._repo[name] = self._sparse(self.long, self.coords)
                    return self._repo[name]
                else:
                    return self._repo[name]
            else:
                assert self.long is not None
                self._repo[name] = self._sparse(self.long, self.coords)
                return self._repo[name]
        else:
            return self._repo[name]

    @property
    def data(self) -> Union[np.ndarray, 'sp.COO']:
        """
        Get the underlying data object of the Array.

        Returns:
            The dense or sparse data object.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> arr = Array(data=long, coords=coords)
            >>> arr.data
            array([[10,  0],
                   [ 0, 20]])

            ```
        """
        if self.data_type == 'sparse':
            return self.sparse
        elif self.data_type == 'dense':
            return self.dense
        else:
            raise ValueError(
                f"data_type must be 'sparse' or 'dense', not {self.data_type}")

    def _shape(self, coords):
        """
        Calculate the shape of the Array based on the provided coordinates.

        Args:
            coords: The coordinates of the Array.

        Returns:
            A list representing the shape of the Array.
        """
        return [coords[dim].size for dim in coords]

    @property
    def shape(self) -> List[int]:
        """
        Get the shape of the Array.

        Returns:
            A list representing the shape of the Array.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> arr = Array(data=long, coords=coords)
            >>> arr.shape
            [2, 2]

            ```
        """
        return self._shape(self.coords)

    def _capacity(self, coords):
        """
        Calculate the capacity of the Array based on the provided coordinates.

        Args:
            coords: The coordinates of the Array.

        Returns:
            The total number of elements the Array can hold.
        """
        return int(np.prod(self._shape(coords)))

    @property
    def capacity(self) -> int:
        """
        Get the capacity of the Array.

        Returns:
            The total number of elements the Array can hold.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> arr = Array(data=long, coords=coords)
            >>> arr.capacity
            4

            ```
        """
        return self._capacity(self.coords)

    @property
    def dims(self) -> List[str]:
        """
        Get the dimensions of the Array.

        Returns:
            A list of dimension names.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long)
            >>> arr.dims
            ['dim1', 'dim2']
            >>> arr.coords
            {'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}

            ```
        """
        return list(self.coords)

    def dindex(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Get the dense index of the Array.

        Returns:
            A dictionary representing the dense index of the Array.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b']), 'dim2': np.array([1, 2])}
            >>> arr = Array(data=long, coords=coords)
            >>> for dim, idx in arr.dindex():
            ...     print(dim, idx)
            dim1 ['a' 'a' 'b' 'b']
            dim2 [1 2 1 2]

            ```
        """
        if len(self.coords) == 0:
            yield from ()
        else:
            arrays = np.unravel_index(np.arange(self._capacity(self.coords)), self._shape(self.coords))
            for dim, idx in zip(self.coords, arrays):
                yield dim, self.coords[dim][idx]

    def sindex(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Get the sparse index of the Array.

        Returns:
            A dictionary representing the sparse index of the Array.

        Example:
            ```python
            >>> data = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
            >>> arr = Array(data=data)
            >>> for dim, idx in arr.sindex():
            ...     print(dim, idx)
            dim1 ['a' 'b']
            dim2 [1 2]

            ```
        """
        if len(self.coords) == 0:
            yield from ()
        else:
            for dim, idx in zip(self.coords, self.sparse.coords):
                yield dim, self.coords[dim][idx]

    def _filler_and_dtype(self, long_value: np.ndarray, fill_missing: Union[float, int, bool, None]) -> Tuple[Union[float, int, bool], np.dtype]:
        """
        Determine the filler value and data type based on the Long object and fill_missing value.

        Args:
            long: The Long object.
            fill_missing: The value to use for missing elements.

        Returns:
            A tuple containing the filler value and the data type.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10.0, 20.0])
            >>> arr = Array(data=({},[]), coords={})
            >>> arr._filler_and_dtype(long.value, fill_missing=0.0)
            (0.0, dtype('float64'))

            ```
        """
        if issubclass(long_value.dtype.type, (np.float16, np.float32, np.float64)):
            dtype = long_value.dtype
            if np.isnan(fill_missing) or np.isinf(fill_missing):
                filler = fill_missing
            elif isinstance(fill_missing, float):
                filler = fill_missing
            elif isinstance(fill_missing, (int, bool)):
                filler = float(fill_missing)
            else:
                raise TypeError("fill_missing must be a float, int or bool")
        elif issubclass(long_value.dtype.type, (np.int16, np.int32, np.int64)):
            dtype = long_value.dtype
            if np.isnan(fill_missing) or np.isinf(fill_missing):
                filler = fill_missing
                dtype = float
            elif isinstance(fill_missing, float):
                filler = fill_missing
                dtype = float
            elif isinstance(fill_missing, int):
                filler = fill_missing
            elif isinstance(fill_missing, bool):
                if fill_missing is True:
                    filler = 1
                else:
                    filler = 0
            else:
                raise TypeError("fill_missing must be a float, int or bool")
        elif issubclass(long_value.dtype.type, np.bool_):
            dtype = long_value.dtype
            if np.isnan(fill_missing) or np.isinf(fill_missing):
                filler = fill_missing
                dtype = float
            elif isinstance(fill_missing, float):
                if fill_missing == 0.0:
                    filler = False
                elif fill_missing == 1.0:
                    filler = True
                else:
                    filler = fill_missing
                    dtype = float
            elif isinstance(fill_missing, int):
                if fill_missing == 0:
                    filler = False
                elif fill_missing == 1:
                    filler = True
                else:
                    filler = float(fill_missing)
                    dtype = float
            elif isinstance(fill_missing, bool):
                filler = fill_missing
            else:
                raise TypeError("fill_missing must be a float, int or bool")
        else:
            raise TypeError(
                f"long_value type is not recognized. Currently {fill_missing=} and {long_value.dtype=} and {long_value.dtype.type=}")
        return filler, dtype

    def _dense(self, long: Long, coords: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert a Long object to a dense array.

        Args:
            long: The Long object to convert.
            coords: The coordinates of the array.

        Returns:
            A dense numpy array.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': np.array(['a', 'b'], dtype=object), 'dim2': np.array([1, 2], dtype=int)}
            >>> arr = Array(data=({},[]), coords={})
            >>> arr._dense(long, coords)
            array([[10,  0],
                   [ 0, 20]])

            ```
        """
        if len(coords) == 0:
            return long.value
        long_stack = np.vstack([np.argsort(coords[dim])[np.searchsorted(
            coords[dim], long.index[dim], sorter=np.argsort(coords[dim]))] for dim in coords])
        shape = self._shape(coords)
        indexes = np.ravel_multi_index(long_stack, shape)
        # Check for duplicate indexes
        self._check_duplicate_indexes(
            indexes, dims=list(coords.keys()), coords=coords)
        capacity = self._capacity(coords)
        filler, dtype = self._filler_and_dtype(long.value, self.fill_value)
        flatten_dense = np.empty((capacity,), dtype=dtype)
        flatten_dense[:] = filler
        flatten_dense[indexes] = long.value.astype(dtype)
        nd_dense = flatten_dense.view().reshape(shape)
        return nd_dense

    def _dense_to_long(self, dense: np.ndarray, coords: Dict[str, np.ndarray]) -> Long:
        """
        Convert a dense array to a Long object.

        Args:
            dense: The dense array to convert.
            coords: The coordinates of the array.

        Returns:
            A Long object.


        Example:
            ```python
            >>> dense = np.array([[10, 0], [0, 20]])
            >>> coords = {'dim1': np.array(['a', 'b'], dtype=object), 'dim2': np.array([1, 2], dtype=int)}
            >>> arr = Array(data=({},[]), coords={})
            >>> arr._dense_to_long(dense, coords)
            Long(index={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, value=array([10., 20.]))

            ```
        """
        if issubclass(dense.dtype.type, (np.int16, np.int32, np.int64)):
            dense = dense.astype(float)
        if len(coords) == 0 and dense.ndim == 1:
            return Long(index={}, value=dense)
        arrays = np.unravel_index(
            np.arange(self._capacity(coords)), self._shape(coords))
        index = {dim: coords[dim][idx] for dim, idx in zip(coords, arrays)}
        long = Long(index=index, value=dense.reshape(dense.size))
        return self._reorder_long(long, list(coords), self.keep_zeros)

    def _sparse(self, long: Long, coords: Dict[str, np.ndarray]) -> 'sp.COO':
        """
        Convert a Long object to a sparse COO array.

        Args:
            long: The Long object to convert.
            coords: The coordinates of the array.

        Returns:
            A sparse COO array.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10., 20.])
            >>> coords = {'dim1': np.array(['a', 'b'], dtype=object), 'dim2': np.array([1, 2], dtype=int)}
            >>> arr = Array(data=({},[]), coords={})
            >>> arr._sparse(long, coords)
            <COO: shape=(2, 2), dtype=float64, nnz=2, fill_value=0.0>

            ```
        """
        if len(coords) == 0:
            return sp.COO(data=long.value, coords=[0], shape=(1,))
        long_stack = np.vstack([np.argsort(coords[dim])[np.searchsorted(
            coords[dim], long.index[dim], sorter=np.argsort(coords[dim]))] for dim in coords])
        shape = self._shape(coords)
        indexes = np.ravel_multi_index(long_stack, shape)
        # Check for duplicate indexes
        self._check_duplicate_indexes(indexes, dims=list(coords.keys()), coords=coords)
        filler, dtype = self._filler_and_dtype(long.value, self.fill_value)
        return sp.COO(coords=long_stack, data=long.value.astype(dtype), shape=shape, fill_value=filler)

    def _sparse_to_long(self, sparse: 'sp.COO', coords: Dict[str, np.ndarray]) -> Long:
        """
        Convert a sparse COO array to a Long object.

        Args:
            sparse: The sparse COO array to convert.
            coords: The coordinates of the array.

        Returns:
            A Long object.

        Example:
            ```python
            >>> sparse = sp.COO(data=[10, 20], coords=[[0, 1], [0, 1]], shape=(2, 2))
            >>> coords = {'dim1': np.array(['a', 'b'], dtype=object), 'dim2': np.array([1, 2], dtype=int)}
            >>> arr = Array(data=({},[]), coords={})
            >>> arr._sparse_to_long(sparse, coords)
            Long(index={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, value=array([10, 20]))

            ```
        """
        index = {dim: coords[dim][idx]
                 for dim, idx in zip(coords, sparse.coords)}
        return Long(index=index, value=sparse.data)

    @staticmethod
    def _reorder(self_long: Long, self_coords: Dict[str, np.ndarray], reorder: List[str] = None) -> Dict[str, Union[Long, Dict[str, np.ndarray]]]:
        """
        Reorder the dimensions of a Long object and its coordinates.

        Args:
            self_long: The Long object to reorder.
            self_coords: The coordinates of the array.
            reorder: The desired order of dimensions.

        Returns:
            A dictionary containing the reordered Long object and coordinates.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> Array._reorder(long, coords, reorder=['dim2', 'dim1'])
            {'data': Long(index={'dim2': array([1, 2]), 'dim1': array(['a', 'b'], dtype=object)}, value=array([10, 20])), 'coords': {'dim2': [1, 2], 'dim1': ['a', 'b']}}

            ```
        """
        assert reorder is not None, "order must be provided"
        assert set(reorder) == set(
            self_long.dims), "order must be equal to self.dims, the order can be different, though"
        if tuple(self_long.dims) == tuple(reorder):
            return dict(data=self_long, coords=self_coords)
        coords = {k: self_coords[k] for k in reorder}
        long = self_long[reorder, :]
        return dict(data=long, coords=coords)

    def reorder(self, reorder: List[str] = None) -> 'Array':
        """
        Reorder the dimensions of the Array.

        Args:
            reorder: The desired order of dimensions.

        Returns:
            A new Array with the reordered dimensions.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=long, coords=coords)
            >>> arr.reorder(reorder=['dim2', 'dim1'])
            Array(data=array([[10,  0],
                   [ 0, 20]]), coords={'dim2': array([1, 2]), 'dim1': array(['a', 'b'], dtype=object)})

            ```
        """
        return Array(**self._reorder(self.long, self.coords, reorder))

    @staticmethod
    def _order_with_preference(dims: List[str], preferred_order: List[str] = None) -> List[str]:
        """
        Order the dimensions based on the preferred order.

        Args:
            dims: The list of dimensions to order.
            preferred_order: The preferred order of dimensions.

        Returns:
            The ordered list of dimensions.

        Example:
            ```python
            >>> dims = ['dim1', 'dim2', 'dim3']
            >>> preferred_order = ['dim2', 'dim3']
            >>> Array._order_with_preference(dims, preferred_order)
            ['dim2', 'dim3', 'dim1']
        """
        if preferred_order is None:
            return dims
        else:
            ordered = []
            disordered = dims[:]
            for dim in preferred_order:
                if dim in disordered:
                    ordered.append(dim)
                    disordered.remove(dim)
            ordered.extend(disordered)
            return ordered

    def _union_dims(self, other: 'Array', preferred_order: List[str] = None) -> List[str]:
        """
        Find the union of dimensions between two arrays. It also performs several checks to ensure the union is valid to perform mathematical operations between arrays.

        Args:
            other: The other array to find the union with.
            preferred_order: The preferred order of dimensions.

        Returns:
            The list of dimensions in the union.

        Example:
            ```python
            >>> long1 = Long(index={'dim1': ['a', 'b']}, value=[10., 20.])
            >>> coords1 = {'dim1': ['a', 'b']}
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> long2 = Long(index={'dim1': ['a', 'a'], 'dim2': [1, 2]}, value=[10., 20.])
            >>> coords2 = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1._union_dims(arr2, preferred_order=['dim1', 'dim2'])
            ['dim1', 'dim2']

            ```
        """
        if set(self.dims) == set(other.dims):
            return self._order_with_preference(self.dims, preferred_order)
        elif len(self.dims) == 0 or len(other.dims) == 0:
            for obj in [self, other]:
                if len(obj.dims) > 0:
                    dims = obj.dims
            return self._order_with_preference(dims, preferred_order)
        elif len(set(self.dims).symmetric_difference(set(other.dims))) > 0:
            common_dims = set(self.dims).intersection(set(other.dims))
            assert len(common_dims) > 0, "At least one dimension must be common"
            uncommon_dims = set(
                self.dims).symmetric_difference(set(other.dims))
            uncommon_self = [dim for dim in self.dims if dim in uncommon_dims]
            uncommon_other = [
                dim for dim in other.dims if dim in uncommon_dims]
            assert not all([len(uncommon_self) > 0, len(uncommon_other) > 0]
                           ), f"It is not allowed to have both arrays with uncommon dims. You can apply .expand in one array before performing this operation. {uncommon_self=} {uncommon_other=}"
            unordered = list(set(self.dims).union(set(other.dims)))
            semi_ordered = self._order_with_preference(
                unordered, preferred_order)
            ordered_common = []
            if preferred_order is None:
                dims = list(common_dims) + list(uncommon_dims)
                return dims
            else:
                for dim in preferred_order:
                    if dim in common_dims:
                        ordered_common.append(dim)
                        common_dims.remove(dim)
                ordered_common.extend(common_dims)
                for dim in ordered_common:
                    if dim in semi_ordered:
                        semi_ordered.remove(dim)
                ordered = ordered_common + semi_ordered
                return ordered

    def _union_coords(self, other: 'Array', uniondims: List[str]) -> Tuple[bool, bool, Dict[str, np.ndarray]]:
        """
        Find the union of coordinates between two arrays.

        Args:
            other: The other array to find the union with.
            uniondims: The list of dimensions in the union.

        Returns:
            A tuple containing boolean flags indicating if the coordinates are the same for each array, and the union of coordinates.

        Example:
            ```python
            >>> long1 = Long(index={'dim1': ['a', 'b']}, value=[10., 20.])
            >>> coords1 = {'dim1': ['a', 'b']}
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> long2 = Long(index={'dim1': ['a', 'a'], 'dim2': [1, 2]}, value=[10., 20.])
            >>> coords2 = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> uniondims = arr1._union_dims(arr2, preferred_order=['dim1', 'dim2'])
            >>> uniondims
            ['dim1', 'dim2']
            >>> arr1._union_coords(arr2, uniondims)
            (True, False, {'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """

        coords = {}
        self_coords_bool = []
        other_coords_bool = []
        for dim in uniondims:
            if dim in self.coords:
                if dim in other.coords:
                    if self.coords[dim].size == other.coords[dim].size:
                        if all(self.coords[dim] == other.coords[dim]):
                            self_coords_bool.append(True)
                            other_coords_bool.append(True)
                            coords[dim] = self.coords[dim]
                        else:
                            coords[dim] = np.union1d(
                                self.coords[dim], other.coords[dim])
                            if coords[dim].size == self.coords[dim].size:
                                if all(coords[dim] == self.coords[dim]):
                                    self_coords_bool.append(True)
                                else:
                                    self_coords_bool.append(False)
                            else:
                                self_coords_bool.append(False)
                            if coords[dim].size == other.coords[dim].size:
                                if all(coords[dim] == other.coords[dim]):
                                    other_coords_bool.append(True)
                                else:
                                    other_coords_bool.append(False)
                            else:
                                other_coords_bool.append(False)
                    elif set(self.coords[dim]).issubset(set(other.coords[dim])):
                        self_coords_bool.append(False)
                        other_coords_bool.append(True)
                        coords[dim] = other.coords[dim]
                    elif set(other.coords[dim]).issubset(set(self.coords[dim])):
                        self_coords_bool.append(True)
                        other_coords_bool.append(False)
                        coords[dim] = self.coords[dim]
                    else:
                        self_coords_bool.append(False)
                        other_coords_bool.append(False)
                        coords[dim] = np.union1d(
                            self.coords[dim], other.coords[dim])
                else:
                    self_coords_bool.append(True)
                    coords[dim] = self.coords[dim]
            elif dim in other.coords:
                other_coords_bool.append(True)
                coords[dim] = other.coords[dim]
            else:
                raise Exception(f"Dimension {dim} not found in either arrays")
        self_coords_bool_ = all(self_coords_bool)
        other_coords_bool_ = all(other_coords_bool)
        return (self_coords_bool_, other_coords_bool_, coords)

    def _get_raw_dense(self, uniondims: List[str], unioncoords: Dict[str, np.ndarray], coords_bool: bool) -> np.ndarray:
        """
        Get the raw dense array based on the union dimensions and coordinates.

        Args:
            uniondims: The list of dimensions in the union.
            unioncoords: The union of coordinates.
            coords_bool: A boolean flag indicating if the coordinates are the same.

        Returns:
            The raw dense array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> uniondims = ['dim1', 'dim2']
            >>> unioncoords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> coords_bool = True
            >>> arr._get_raw_dense(uniondims, unioncoords, coords_bool)
            array([[10,  0],
                   [ 0, 20]])

            ```
        """
        self_dims = [d for d in uniondims if d in self.dims]
        if coords_bool:
            if tuple(self.dims) == tuple(self_dims):
                self_raw_dense = self.dense
                return self_raw_dense
        self_coords = {d: unioncoords[d] for d in self_dims}
        self_raw_dense = self._dense(self.long, self_coords)
        return self_raw_dense

    def _get_raw_sparse(self, uniondims: List[str], unioncoords: Dict[str, np.ndarray], coords_bool: bool) -> 'sp.COO':
        """
        Get the raw sparse array based on the union dimensions and coordinates.

        Args:
            uniondims: The list of dimensions in the union.
            unioncoords: The union of coordinates.
            coords_bool: A boolean flag indicating if the coordinates are the same.

        Returns:
            The raw sparse array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10., 20.])
            >>> arr = Array(data=long, coords=coords)
            >>> uniondims = ['dim1', 'dim2']
            >>> unioncoords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> coords_bool = True
            >>> arr._get_raw_sparse(uniondims, unioncoords, coords_bool)
            <COO: shape=(2, 2), dtype=float64, nnz=2, fill_value=0.0>

            ```
        """
        self_dims = [d for d in uniondims if d in self.dims]
        if coords_bool:
            if tuple(self.dims) == tuple(self_dims):
                self_raw_sparse = self.sparse
                return self_raw_sparse
        self_coords = {d: unioncoords[d] for d in self_dims}
        self_raw_sparse = self._sparse(self.long, self_coords)
        return self_raw_sparse

    def _pre_operation_with_array(self, other: 'Array') -> Tuple[Union[np.ndarray, 'sp.COO'], Union[np.ndarray, 'sp.COO'], Dict[str, np.ndarray]]:
        """
        Perform pre-operation steps with another array.

        Args:
            other: The other array to perform the operation with.

        Returns:
            A tuple containing the raw arrays and the union of coordinates.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[30, 40])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1._pre_operation_with_array(arr2)
            (array([[10,  0],
                   [ 0, 20]]), array([[30,  0],
                   [ 0, 40]]), {'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        uniondims = self._union_dims(other, preferred_order=self.order)
        self_coords_bool, other_coords_bool, unioncoords = self._union_coords(
            other, uniondims)
        if self.data_type == "sparse" and other.data_type == "sparse":
            self_raw_sparse = self._get_raw_sparse(
                uniondims, unioncoords, self_coords_bool)
            other_raw_sparse = other._get_raw_sparse(
                uniondims, unioncoords, other_coords_bool)
            return self_raw_sparse.T, other_raw_sparse.T, unioncoords
        elif self.data_type == "dense" and other.data_type == "dense":
            self_raw_dense = self._get_raw_dense(
                uniondims, unioncoords, self_coords_bool)
            other_raw_dense = other._get_raw_dense(
                uniondims, unioncoords, other_coords_bool)
            return self_raw_dense.T, other_raw_dense.T, unioncoords
        elif self.data_type == "sparse" and other.data_type == "dense":
            self_raw_sparse = self._get_raw_sparse(
                uniondims, unioncoords, self_coords_bool)
            other_raw_sparse = other._get_raw_sparse(
                uniondims, unioncoords, other_coords_bool)
            return self_raw_sparse.T, other_raw_sparse.T, unioncoords
        elif self.data_type == "dense" and other.data_type == "sparse":
            self_raw_dense = self._get_raw_dense(
                uniondims, unioncoords, self_coords_bool)
            other_raw_dense = other._get_raw_dense(
                uniondims, unioncoords, other_coords_bool)
            return self_raw_dense.T, other_raw_dense.T, unioncoords
        else:
            raise Exception("data_type must be 'sparse' or 'dense'")

    def _post_operation(self, resulting_array: Union[np.ndarray, 'sp.COO'], coords: Dict[str, np.ndarray]) -> 'Array':
        """
        Perform post-operation steps and create a new Array object.

        Args:
            resulting_array: The resulting array from the operation.
            coords: The coordinates of the resulting array.

        Returns:
            A new Array object with the resulting array and coordinates.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[30, 40])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> resulting_array = arr1.data + arr2.data
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr1._post_operation(resulting_array, coords)
            Array(data=array([[40.,  0.],
                   [ 0., 60.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if len(coords) == 0:
            return Array(data=({}, resulting_array), coords={})
        return Array(data=resulting_array, coords=coords)

    def _operation(self, self_array: Union[np.ndarray, 'sp.COO', 'cp.ndarray', 'torch.Tensor'], other_array: Union[np.ndarray, 'sp.COO', 'cp.ndarray', 'torch.Tensor'], operation: str) -> Union[np.ndarray, 'sp.COO']:
        """
        Perform a math operation on two arrays.

        Args:
            self_array: The first array.
            other_array: The second array.
            operation: The operation to perform.

        Returns:
            The result of the operation.
        """
        if isinstance(self_array, np.ndarray) and isinstance(other_array, np.ndarray):
            if self.gpu_backend is not None:
                if self.gpu_backend == 'cupy':
                    mempool = cp.get_default_memory_pool()
                    with cp.cuda.Device(self.gpu_backend_device):
                        self_array = cp.array(self_array)
                        other_array = cp.array(other_array)
                        result = getattr(self_array, operation)(other_array)
                        np_array = result.get()
                    del result, self_array, other_array
                    mempool.free_all_blocks()
                    return np_array
                elif self.gpu_backend == 'pytorch':
                    with torch.cuda.device(self.gpu_backend_device):
                        self_array = torch.tensor(self_array)
                        other_array = torch.tensor(other_array)
                        result = getattr(self_array, operation)(other_array)
                        np_array = result.cpu().numpy()
                        del result, self_array, other_array
                    return np_array
            else:
                return getattr(self_array, operation)(other_array)
        elif isinstance(self_array, sp.COO) and isinstance(other_array, sp.COO):
            return getattr(self_array, operation)(other_array)
        else:
            raise Exception("Invalid data type for 'data' property")

    def __add__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Add the Array with another Array or a number.

        Args:
            other: The other Array or number to add.

        Returns:
            A new Array object with the result of the addition.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[30, 40])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1 + arr2
            Array(data=array([[40.,  0.],
                   [ 0., 60.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 + 5
            Array(data=array([[15.,  5.],
                   [ 5., 25.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data + other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self._operation(self_arr, other_arr, '__add__')
            return self._post_operation(arr.T, coords)

    def __mul__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Multiply the Array with another Array or a number.

        Args:
            other: The other Array or number to multiply.

        Returns:
            A new Array object with the result of the multiplication.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[30, 40])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1 * arr2
            Array(data=array([[300.,   0.],
                   [  0., 800.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 * 5
            Array(data=array([[ 50.,   0.],
                   [  0., 100.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data * other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self._operation(self_arr, other_arr, '__mul__')
            return self._post_operation(arr.T, coords)

    def __sub__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Subtract another Array or a number from the Array.

        Args:
            other: The other Array or number to subtract.

        Returns:
            A new Array object with the result of the subtraction.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[30, 40])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1 - arr2
            Array(data=array([[-20.,   0.],
                   [  0., -20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 - 5
            Array(data=array([[ 5., -5.],
                   [-5., 15.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data - other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self._operation(self_arr, other_arr, '__sub__')
            return self._post_operation(arr.T, coords)

    def __truediv__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Divide the Array by another Array or a number.

        Args:
            other: The other Array or number to divide by.

        Returns:
            A new Array object with the result of the division.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a','a', 'b', 'b'], 'dim2': [1, 2, 1, 2]}, value=[30, 40, 50, 60])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1 / arr2
            Array(data=array([[0.33333333, 0.        ],
                   [0.        , 0.33333333]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 / 5
            Array(data=array([[2., 0.],
                   [0., 4.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data / other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self._operation(self_arr, other_arr, '__truediv__')
            return self._post_operation(arr.T, coords)

    def __radd__(self, other: Union[int, float]) -> 'Array':
        """
        Add a number to the Array (reverse addition).

        Args:
            other: The number to add.

        Returns:
            A new Array object with the result of the addition.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> 5 + arr
            Array(data=array([[15.,  5.],
                   [ 5., 25.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data + other
            return self._post_operation(arr, self.coords)

    def __rmul__(self, other: Union[int, float]) -> 'Array':
        """
        Multiply a number with the Array (reverse multiplication).

        Args:
            other: The number to multiply.

        Returns:
            A new Array object with the result of the multiplication.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> 5 * arr
            Array(data=array([[ 50.,   0.],
                   [  0., 100.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data * other
            return self._post_operation(arr, self.coords)

    def __rsub__(self, other: Union[int, float]) -> 'Array':
        """
        Subtract the Array from a number (reverse subtraction).

        Args:
            other: The number to subtract from.

        Returns:
            A new Array object with the result of the subtraction.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> 5 - arr
            Array(data=array([[ -5.,   5.],
                   [  5., -15.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = -self.data + other
            return self._post_operation(arr, self.coords)

    def __rtruediv__(self, other: Union[int, float]) -> 'Array':
        """
        Divide a number by the Array (reverse division).

        Args:
            other: The number to divide.

        Returns:
            A new Array object with the result of the division.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'a', 'b', 'b'], 'dim2': [1, 2, 1, 2]}, value=[10, 20, 30, 40])
            >>> arr = Array(data=long, coords=coords)
            >>> 100 / arr
            Array(data=array([[10.        ,  5.        ],
                   [ 3.33333333,  2.5       ]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other / self.data
            return self._post_operation(arr, self.coords)

    def __neg__(self) -> 'Array':
        """
        Negate the Array.

        Returns:
            A new Array object with the negated values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> -arr
            Array(data=array([[-10.,   0.],
                   [  0., -20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self._post_operation(-self.data, self.coords)

    def __pos__(self) -> 'Array':
        """
        Apply the unary positive operator to the Array.

        Returns:
            A new Array object with the same values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> +arr
            Array(data=array([[10.,  0.],
                   [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self._post_operation(+self.data, self.coords)

    def __eq__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Check equality between the Array and another Array or a number.

        Args:
            other: The other Array or number to compare.

        Returns:
            A new Array object with boolean values indicating equality.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [2, 1]}, value=[20, 10])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1 == arr2
            Array(data=array([[False, False],
                   [False, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 == 10
            Array(data=array([[ True, False],
                   [False, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data == other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr == other_arr
            return self._post_operation(arr.T, coords)

    def __ne__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Check inequality between the Array and another Array or a number.

        Args:
            other: The other Array or number to compare.

        Returns:
            A new Array object with boolean values indicating inequality.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data1 = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
            >>> arr1 = Array(data=data1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data2 = ({'dim1': ['a', 'b'], 'dim2': [2, 1]}, [20, 10])
            >>> arr2 = Array(data=data2, coords=coords2)
            >>> arr1 != arr2
            Array(data=array([[ True,  True],
                   [ True,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 != 10
            Array(data=array([[False,  True],
                   [ True,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data != other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr != other_arr
            return self._post_operation(arr.T, coords)

    def __lt__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Check if the Array is less than another Array or a number.

        Args:
            other: The other Array or number to compare.

        Returns:
            A new Array object with boolean values indicating less than.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data1 = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
            >>> arr1 = Array(data=data1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data2 = ({'dim1': ['a', 'b'], 'dim2': [2, 1]}, [20, 10])
            >>> arr2 = Array(data=data2, coords=coords2)
            >>> arr1 < arr2
            Array(data=array([[False,  True],
                   [ True, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 < 15
            Array(data=array([[ True,  True],
                   [ True, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data < other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr < other_arr
            return self._post_operation(arr.T, coords)

    def __rlt__(self, other: Union[int, float]) -> 'Array':
        """
        Check if a number is less than the Array (reverse less than).

        Args:
            other: The number to compare.

        Returns:
            A new Array object with boolean values indicating less than.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
            >>> arr = Array(data=data, coords=coords)
            >>> 5 < arr
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other < self.data
            return self._post_operation(arr, self.coords)

    def __le__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Check if the Array is less than or equal to another Array or a number.

        Args:
            other: The other Array or number to compare.

        Returns:
            A new Array object with boolean values indicating less than or equal to.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data1 = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
            >>> arr1 = Array(data=data1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [2, 1]}, value=[20, 10])
            >>> arr2 = Array(data=long2, coords=coords2)
            >>> arr1 <= arr2
            Array(data=array([[False,  True],
                   [ True, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 <= 10
            Array(data=array([[ True,  True],
                   [ True, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data <= other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr <= other_arr
            return self._post_operation(arr.T, coords)

    def __rle__(self, other: Union[int, float]) -> 'Array':
        """
        Check if a number is less than or equal to the Array (reverse less than or equal to).

        Args:
            other: The number to compare.

        Returns:
            A new Array object with boolean values indicating less than or equal to.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
            >>> arr = Array(data=data, coords=coords)
            >>> 10 <= arr
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other <= self.data
            return self._post_operation(arr, self.coords)

    def __gt__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Check if the Array is greater than another Array or a number.

        Args:
            other: The other Array or number to compare.

        Returns:
            A new Array object with boolean values indicating greater than.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr1 = Array(data=long1, coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> data2 = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [20, 10])
            >>> arr2 = Array(data=data2, coords=coords2)
            >>> arr1 > arr2
            Array(data=array([[False, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 > 15
            Array(data=array([[False, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data > other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr > other_arr
            return self._post_operation(arr.T, coords)

    def __rgt__(self, other: Union[int, float]) -> 'Array':
        """
        Check if a number is greater than the Array (reverse greater than).

        Args:
            other: The number to compare.

        Returns:
            A new Array object with boolean values indicating greater than.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> dense = np.array([[10, 20], [20, 10]])
            >>> arr = Array(data=dense, coords=coords)
            >>> 15 > arr
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other > self.data
            return self._post_operation(arr, self.coords)

    def __ge__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Check if the Array is greater than or equal to another Array or a number.

        Args:
            other: The other Array or number to compare.

        Returns:
            A new Array object with boolean values indicating greater than or equal to.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[20, 10])
            >>> arr1 = Array(data=long1, coords=coords)
            >>> arr2 = Array(data=long2, coords=coords)
            >>> arr1 >= arr2
            Array(data=array([[False,  True],
                   [ True,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 >= 10
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data >= other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr >= other_arr
            return self._post_operation(arr.T, coords)

    def __rge__(self, other: Union[int, float]) -> 'Array':
        """
        Check if a number is greater than or equal to the Array (reverse greater than or equal to).

        Args:
            other: The number to compare.

        Returns:
            A new Array object with boolean values indicating greater than or equal to.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> 10 >= arr
            Array(data=array([[ True,  True],
                   [ True, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        arr = other >= self.data
        return self._post_operation(arr, self.coords)

    def __and__(self, other: Union[bool, 'Array']) -> 'Array':
        """
        Perform element-wise logical AND operation between the Array and another Array or a boolean.

        Args:
            other: The other Array or boolean to perform the operation with.

        Returns:
            A new Array object with the result of the logical AND operation.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[20, 10])
            >>> arr1 = Array(data=long1, coords=coords)
            >>> arr2 = Array(data=long2, coords=coords)
            >>> arr1 & arr2
            Array(data=array([[0., 0.],
                   [0., 0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 & True
            Array(data=array([[0., 0.],
                   [0., 0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data & other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr & other_arr
            return self._post_operation(arr.T, coords)

    def __rand__(self, other: bool) -> 'Array':
        """
        Perform element-wise logical AND operation between a boolean and the Array (reverse AND).

        Args:
            other: The boolean to perform the operation with.

        Returns:
            A new Array object with the result of the logical AND operation.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long)
            >>> True & arr
            Array(data=array([[0., 0.],
                   [0., 0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, bool):
            arr = other & self.data
            return self._post_operation(arr, self.coords)

    def __or__(self, other: Union[int, float, 'Array']) -> 'Array':
        """
        Perform element-wise logical OR operation between the Array and another Array or a number.

        Args:
            other: The other Array or number to perform the operation with.

        Returns:
            A new Array object with the result of the logical OR operation.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[20, 10])
            >>> arr1 = Array(data=long1, coords=coords)
            >>> arr2 = Array(data=long2, coords=coords)
            >>> arr1 | arr2
            Array(data=array([[30.,  0.],
                   [ 0., 30.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> arr1 | 0
            Array(data=array([[10.,  0.],
                   [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data | other
            return self._post_operation(arr, self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation_with_array(other)
            arr = self_arr | other_arr
            return self._post_operation(arr.T, coords)

    def __ror__(self, other: bool) -> 'Array':
        """
        Perform element-wise logical OR operation between a boolean and the Array (reverse OR).

        Args:
            other: The boolean to perform the operation with.

        Returns:
            A new Array object with the result of the logical OR operation.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long)
            >>> True | arr
            Array(data=array([[11.,  1.],
                   [ 1., 21.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, bool):
            arr = other | self.data
            return self._post_operation(arr, self.coords)

    def __invert__(self) -> 'Array':
        """
        Perform element-wise logical NOT operation on the Array.

        Returns:
            A new Array object with the result of the logical NOT operation.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> ~arr
            Array(data=array([[-11.,  -1.],
                   [ -1., -21.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self._post_operation(~self.data, coords=self.coords)

    def __bool__(self) -> bool:
        """
        Raise a ValueError when trying to convert an Array to a boolean. Useful to warn the user to implement .all() or .any() instead of bool(array).

        Raises:
            ValueError: Cannot convert an Array with more than one element to a boolean.
        """
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous. Use Array.any() or Array.all()")

    def any(self) -> bool:
        """
        Check if any element in the Array is True.

        Returns:
            True if any element is True, False otherwise.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> arr.any()
            True

            ```
        """
        return self.data.any()

    def all(self) -> bool:
        """
        Check if all elements in the Array are True.

        Returns:
            True if all elements are True, False otherwise.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> arr.all()
            False

            ```
        """
        return self.data.all()

    def to_dict(self, dense: bool = False) -> Dict[str, np.ndarray]:
        """
        Convert the Array to a dictionary.

        Args:
            dense: Whether to convert the dense representation of the Array.

        Returns:
            A dictionary representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10., 20.])
            >>> arr = Array(data=long, coords=coords)
            >>> arr.to_dict(dense=True)
            {'dim1': array(['a', 'a', 'b', 'b'], dtype=object), 'dim2': array([1, 2, 1, 2]), 'value': array([10.,  0.,  0., 20.])}

            ```
        """
        if self.data_type == 'dense':
            if dense:
                value = self.dense.flatten()
                array_dict = dict(self.dindex())
                array_dict['value'] = value
                return array_dict
            else:
                raw_value = self.dense.flatten()
                filler, dtype = self._filler_and_dtype(raw_value, self.fill_value)
                value = raw_value.astype(dtype)
                non_missing_mask = value != filler
                array_dict = {dim: index[non_missing_mask] for dim, index in self.dindex()}
                array_dict['value'] = value[non_missing_mask]
                return array_dict
        elif self.data_type == 'sparse':
            if dense:
                value = self.todense().flatten()
                array_dict = dict(self.dindex())
                array_dict['value'] = value
                return array_dict
            else:
                value = self.sparse.data
                array_dict = dict(self.sindex())
                array_dict['value'] = value
                return array_dict

    def to_pandas(self, dense: bool = None) -> 'pd.DataFrame':
        """
        Convert the Array to a pandas DataFrame.

        Args:
            dense: Whether to convert the Array to a dense DataFrame. If None, use the default setting.

        Returns:
            A pandas DataFrame representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> df = arr.to_pandas(dense=True)
            >>> df
              dim1  dim2  value
            0    a     1     10
            1    a     2      0
            2    b     1      0
            3    b     2     20

            ```
        """
        dataframe_as_dense = True if self.dataframe_as == "dense" else False
        sparse_to_dense = dataframe_as_dense if dense is None else dense
        return pd.DataFrame(self.to_dict(dense=sparse_to_dense))

    def to_polars(self, dense: bool = None) -> 'pl.DataFrame':
        """
        Convert the Array to a polars DataFrame.

        Args:
            dense: Whether to convert the Array to a dense DataFrame. If None, use the default setting.

        Returns:
            A polars DataFrame representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> df = arr.to_polars(dense=False)
            >>> df
            shape: (2, 3)
            ┌──────┬──────┬───────┐
            │ dim1 ┆ dim2 ┆ value │
            │ ---  ┆ ---  ┆ ---   │
            │ str  ┆ i32  ┆ i64   │
            ╞══════╪══════╪═══════╡
            │ a    ┆ 1    ┆ 10    │
            │ b    ┆ 2    ┆ 20    │
            └──────┴──────┴───────┘

            ```
        """
        dataframe_as_dense = True if self.dataframe_as == "dense" else False
        sparse_to_dense = dataframe_as_dense if dense is None else dense
        return pl.from_dict(self.to_dict(dense=sparse_to_dense))

    def to_dataframe(self, dense: bool = None, with_: str = 'pandas') -> Union['pd.DataFrame', 'pl.DataFrame']:
        """
        Convert the Array to a DataFrame using the specified library.

        Args:
            dense: Whether to convert the Array to a dense DataFrame. If None, use the default setting.
            with_: The library to use for creating the DataFrame. Can be 'pandas' or 'polars'.

        Returns:
            A DataFrame representing the Array, created using the specified library.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> df = arr.to_dataframe(dense=False, with_='pandas')
            >>> df
              dim1  dim2  value
            0    a     1     10
            1    b     2     20

            ```
        """
        assert with_ in ['pandas', 'polars']
        if with_ == "pandas":
            return self.to_pandas(dense=dense)
        elif with_ == "polars":
            return self.to_polars(dense=dense)

    def to_arrow(self) -> 'pa.Table':
        """
        Convert the Array to an Apache Arrow Table.

        Returns:
            An Apache Arrow Table representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> table = arr.to_arrow()
            >>> table
            pyarrow.Table
            dim1: string
            dim2: int32
            value: int64
            ----
            dim1: [["a","b"]]
            dim2: [[1,2]]
            value: [[10,20]]

            ```
        """
        table = pa.Table.from_pydict(self.to_dict())
        custom_meta_key = 'karray'
        custom_metadata = {'coords': {dim: self.coords[dim].tolist() for dim in self.coords}}
        custom_meta_json = json.dumps(custom_metadata)
        existing_meta = table.schema.metadata if table.schema.metadata is not None else {}
        combined_meta = {custom_meta_key.encode(): custom_meta_json.encode(), **existing_meta}
        return table.replace_schema_metadata(combined_meta)

    def to_feather(self, path: str) -> None:
        """
        Save the Array to a Feather file.

        Args:
            path: The path to save the Feather file.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> arr.to_feather('tests/data/array.feather')

            ```
        """
        table = self.to_arrow()
        feather.write_feather(table, path)
        return None

    def to_csv(self, path: str) -> None:
        """
        Save the Array to a CSV file.

        Args:
            path: The path to save the CSV file.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> arr.to_csv('tests/data/array.csv')

            ```
        """
        table = self.to_arrow()
        table.to_pandas().to_csv(path, index=False)
        return None

    def shrink(self, **kwargs: Union[List[Any], np.ndarray]) -> 'Array':
        """
        Shrink the Array by selecting specific elements from the specified dimensions.

        Args:
            **kwargs: Keyword arguments specifying the dimensions and elements to keep.

        Returns:
            A new Array object with the selected elements.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> new_arr = arr.shrink(dim1=['a'], dim2=[1])
            >>> new_arr
            Array(data=array([[10]]), coords={'dim1': array(['a'], dtype=object), 'dim2': array([1])})

            ```
        """
        assert all([kw in self.coords for kw in kwargs]
                   ), "Selected dimension must be in coords"
        assert all([isinstance(kwargs[dim], (list, np.ndarray)) for dim in kwargs]
                   ), "Keeping elements must be contained in lists or np.ndarray"
        assert all([set(kwargs[kw]).issubset(self.coords[kw])
                   for kw in kwargs]), "All keeping elements must be included of coords"
        assert all([len(set(kwargs[kw])) == len(kwargs[kw])
                   for kw in kwargs]), "Keeping elements in list must be unique"
        new_coords = {}
        for dim in self.coords:
            if dim in kwargs:
                new_coords[dim] = _test_type_and_update(kwargs[dim])
            else:
                new_coords[dim] = self.coords[dim]
        long = self.long
        for dim in self.dims:
            if dim in kwargs:
                long = long[dim, kwargs[dim]]
        return Array(data=long, coords=new_coords)

    def add_elem(self, **kwargs: Union[List[Any], np.ndarray]) -> 'Array':
        """
        Add new elements to the specified dimensions of the Array.

        Args:
            **kwargs: Keyword arguments specifying the dimensions and elements to add.

        Returns:
            A new Array object with the added elements.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10., 20.])
            >>> arr = Array(data=long, coords=coords)
            >>> arr
            Array(data=array([[10.,  0.],
                   [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})
            >>> new_arr = arr.add_elem(dim1=['c'], dim2=[3])
            >>> new_arr
            Array(data=array([[10.,  0.,  0.],
                   [ 0., 20.,  0.],
                   [ 0.,  0.,  0.]]), coords={'dim1': array(['a', 'b', 'c'], dtype=object), 'dim2': array([1, 2, 3])})

            ```
        """
        for dim in kwargs:
            assert dim in self.dims, f'dim: {dim} must exist in self.dims: {self.dims}'
        assert all([isinstance(kwargs[dim], (list, np.ndarray, 'pd.DatetimeIndex', 'pd.Categorical')) for dim in kwargs]
                   ), "Keeping elements must be contained in lists, np.ndarray, pd.Categorical or pd.DatetimeIndex"
        coords = {}
        for dim in self.coords:
            if dim in kwargs:
                coords[dim] = np.unique(np.hstack((self.coords[dim], _test_type_and_update(kwargs[dim]))))
            else:
                coords[dim] = self.coords[dim]
        return Array(data=self.long, coords=coords)

    def reduce(self, dim: str, aggfunc: Union[str, Callable] = np.add.reduce) -> 'Array':
        """
        Reduce the Array along a specified dimension using an aggregation function.

        Args:
            dim: The dimension to reduce.
            aggfunc: The aggregation function to apply. Can be a string ('sum', 'mean', 'prod') or a callable.

        Returns:
            A new Array object with the reduced dimension.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> reduced_arr = arr.reduce('dim1', aggfunc='sum')
            >>> reduced_arr
            Array(data=array([10., 20.]), coords={'dim2': array([1, 2])})

            ```
        """
        assert dim in self.dims, f"dim {dim} not in self.dims: {self.dims}"
        if isinstance(aggfunc, str):
            assert aggfunc in [
                'sum', 'mean', 'prod'], "String options for aggfunc can be 'sum', 'mean' or 'prod'"
            if aggfunc == 'sum':
                aggfunc = np.add.reduce
            elif aggfunc == 'mean':
                aggfunc = np.mean
            elif aggfunc == 'prod':
                aggfunc = np.multiply.reduce
        elif isinstance(aggfunc, Callable):
            pass
        ndarray = aggfunc(self.data, axis=self.dims.index(dim))
        dims = [d for d in self.dims if d != dim]
        coords = {k: v for k, v in self.coords.items() if k in dims}
        return self._post_operation(ndarray, coords)

    def _shift_one_dim(self, dim: str, count: int, fill_value: Union[float, None] = None) -> 'Array':
        """
        Shift the Array along a single dimension.

        Args:
            dim: The dimension to shift.
            count: The number of positions to shift. Positive values shift forward, negative values shift backward.
            fill_value: The value to fill the empty positions after shifting. If None, use the default fill value.

        Returns:
            A new Array object with the shifted values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> shifted_arr = arr._shift_one_dim('dim1', count=1, fill_value=0)
            >>> shifted_arr
            Array(data=array([[ 0.,  0.],
                   [10.,  0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if fill_value is None:
            fill_value, dtype = self._filler_and_dtype(self.data, self.fill_value)
            raw_data = self.data.astype(dtype)
        else:
            raw_data = self.data
        ax = self.dims.index(dim)
        data = np.roll(raw_data, shift=count, axis=ax)
        if self.data_type == 'dense':
            if count > 0:
                data.swapaxes(0, ax)[:count] = fill_value
            elif count < 0:
                data.swapaxes(0, ax)[count:] = fill_value
            return self._post_operation(data, self.coords)
        elif self.data_type == 'sparse':
            coo_index = data.coords
            shape = data.shape
            upper = shape[ax]
            if count > 0:
                start = 0
                end = count
            else:
                start = upper + count
                end = upper
            assert _is_in_range(start, end, 0, upper, step=1)
            selected = coo_index[ax]
            sortedidx = np.argsort(selected)
            l = np.searchsorted(selected, start, side='left', sorter=sortedidx)
            r = np.searchsorted(selected, end-1, side='right', sorter=sortedidx)
            inds = sortedidx[l:r]
            coo_value = data.data
            coo_value[inds] = fill_value
            coo = sp.COO(data=coo_value, coords=coo_index, shape=shape)
            return self._post_operation(coo, self.coords)

    def shift(self, fill_value: Union[float, None] = None, **kwargs: int) -> 'Array':
        """
        Shift the Array along specified dimensions.

        Args:
            fill_value: The value to fill the empty positions after shifting. If None, use the default fill value.
            **kwargs: Keyword arguments specifying the dimensions and the number of positions to shift.

        Returns:
            A new Array object with the shifted values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> shifted_arr = arr.shift(dim1=1, dim2=-1, fill_value=0)
            >>> shifted_arr
            Array(data=array([[0., 0.],
                   [0., 0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        assert len(kwargs) > 0
        assert all([dim in self.dims for dim in kwargs])
        assert all([isinstance(kwargs[dim], int) for dim in kwargs])
        obj = self
        for dim in kwargs:
            obj = obj._shift_one_dim(dim=dim, count=kwargs[dim], fill_value=fill_value)
        return obj

    def _roll_one_dim(self, dim: str, count: int) -> 'Array':
        """
        Roll the Array along a single dimension.

        Args:
            dim: The dimension to roll.
            count: The number of positions to roll. Positive values roll forward, negative values roll backward.

        Returns:
            A new Array object with the rolled values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> rolled_arr = arr._roll_one_dim('dim1', count=1)
            >>> rolled_arr
            Array(data=array([[ 0., 20.],
                   [10.,  0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        assert dim in self.dims, f"{dim} not in dims: {self.dims}"
        assert isinstance(count, int), f"{count} must be int"
        ax = self.dims.index(dim)
        data = np.roll(self.data, shift=count, axis=ax)
        return self._post_operation(data, self.coords)

    def roll(self, **kwargs: int) -> 'Array':
        """
        Roll the Array along specified dimensions.

        Args:
            **kwargs: Keyword arguments specifying the dimensions and the number of positions to roll.

        Returns:
            A new Array object with the rolled values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> rolled_arr = arr.roll(dim1=1, dim2=-1)
            >>> rolled_arr
            Array(data=array([[20.,  0.],
                   [ 0., 10.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        assert len(kwargs) > 0, "Must specify at least one dimension to roll"
        assert all([dim in self.dims for dim in kwargs]), f"{kwargs} not in dims: {self.dims}"
        assert all([isinstance(kwargs[dim], int) for dim in kwargs]), f"All values in {kwargs} must be integers"
        obj = self
        for dim in kwargs:
            obj = obj._roll_one_dim(dim=dim, count=kwargs[dim])
        return obj

    def insert(self, **kwargs: Union[np.dtype, type, str, int, float, Dict[str, Union[Dict[str, Any], List[Union[List[str], List[Any]]]]]]) -> 'Array':
        """
        Insert new dimensions into the Array. There are four groups of values that can be inserted:

        - str, int, float

        - dict of dict or dict of list

        - list of str that represents current dimensions

        - np.dtype, type

        Case 1: str, int, float are valid for non-empty arrays. In this case, we insert a new dimension with only one element.
        Case 2 and 3: dict of dict or dict of list are valid for non-empty arrays. In this case, we insert a new dimension which is mapped based on a existing dimension coordinates.
        Case 4: list of str are valid for non-empty arrays. In this case, we insert a new dimension with the concatenation of elements of the corresponding dims in the list.
        Case 5: types are only valid for empty arrays. In order to insert a dimension with a dtype, we need to create an empty array

        Args:
            **kwargs: Keyword arguments specifying the new dimensions and their values.

        Returns:
            A new Array object with the inserted dimensions.

        Example:
            ```python
            >>> # Case 1: str, int, float

            >>> arr = Array(data=({'dim1':['a','b'], 'dim2':[1,2]},[1.0, 2.0]), coords={'dim1':['a','b'], 'dim2':[1,2]})

            >>> new_arr = arr.insert(dim3='c')

            >>> new_arr
            Array(data=array([[[1., 0.],
                    [0., 2.]]]), coords={'dim3': array(['c'], dtype=object), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            >>> # Case 2: dict of dict

            >>> arr = Array(data=({'dim1':['a','b'], 'dim2':[1,2]},[1.0, 2.0]), coords={'dim1':['a','b'], 'dim2':[1,2]})

            >>> new_arr = arr.insert(dim3={'dim1':{'a':'c', 'b':'c'}})

            >>> # Case 3: dict of list

            >>> new_arr = arr.insert(dim3={'dim1':[['a','b'],['c', 'c']]})

            >>> new_arr
            Array(data=array([[[1., 0.],
                    [0., 2.]]]), coords={'dim3': array(['c'], dtype=object), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            >>> # Case 4: list of dims (str)

            >>> arr = Array(data=({'dim1':['a','b'], 'dim2':[1,2]},[1.0, 2.0]), coords={'dim1':['a','b'], 'dim2':[1,2]})

            >>> new_arr = arr.insert(dim3=['dim1', 'dim2'])

            >>> new_arr
            Array(data=array([[[1., 0.],
                    [0., 0.]],
            <BLANKLINE>
                   [[0., 0.],
                    [0., 0.]],
            <BLANKLINE>
                   [[0., 0.],
                    [0., 0.]],
            <BLANKLINE>
                   [[0., 0.],
                    [0., 2.]]]), coords={'dim3': array(['a:1', 'a:2', 'b:1', 'b:2'], dtype=object), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            >>> # Case 5: types

            >>> arr = Array(data=({'dim1':[], 'dim2':[]},[]), coords={'dim1':[], 'dim2':[]})

            >>> new_arr = arr.insert(dim3=np.dtype('int64'))

            >>> new_arr
            Array(data=array([], shape=(0, 0, 0)), coords={'dim3': array([]), 'dim1': array([]), 'dim2': array([])})

            ```
        """
        coords = {}
        for new_dim in kwargs:
            assert new_dim not in self.dims, f"new dimension name '{new_dim}' must not exist in the existing dimensions"
            value = kwargs[new_dim]
            # dtype or type only works for for empty arrays. In order to insert a dimension with a dtype, we need to create an empty array
            if isinstance(value, (np.dtype, type)):
                figures = np.array([], dtype=value)
                coords[new_dim] = _test_type_and_update(figures)
            elif isinstance(value, str):
                coords[new_dim] = np.array([value], dtype=np.object_)
            elif isinstance(value, int):
                coords[new_dim] = np.array([value], dtype=np.int32)
            elif isinstance(value, float):
                coords[new_dim] = np.array([value], dtype=np.float32)
            elif isinstance(value, dict):
                assert len(
                    value) == 1, f"Value associated with '{new_dim}' must be a dict with one key. Got {value}"
                existing_dim = next(iter(value))
                assert isinstance(
                    existing_dim, str), f"Value associated with '{new_dim}' must be a str. Got {type(existing_dim)}"
                assert existing_dim in self.dims, f"Value associated with '{new_dim}' and '{existing_dim}' must be in {self.dims}"
                assert isinstance(value[existing_dim], (
                    dict, list)), f"Value associated with '{new_dim}' and '{existing_dim}' must be a dict or list. Got {type(value[existing_dim])}"
                if isinstance(value[existing_dim], dict):
                    old_dim_items_set = set(value[existing_dim])
                    assert set(
                        self.coords[existing_dim]) == old_dim_items_set, f"All items associated with '{new_dim}' must match elements in .coords['{existing_dim}']. The current mapping dictionary between '{new_dim}' and '{existing_dim}' is matched partially"
                    assert len(value[existing_dim]) == len(
                        old_dim_items_set), f"There are duplicate items in the mapping dict associated with '{new_dim}' and '{existing_dim}'"
                    coords[new_dim] = np.unique(_test_type_and_update(
                        list(value[existing_dim].values())))
                elif isinstance(value[existing_dim], list):
                    assert len(
                        value[existing_dim]) == 2, f"Value associated with '{new_dim}' and '{existing_dim}' must be a list with two items. Got {value[existing_dim]}"
                    old_dim_items_set = set(value[existing_dim][0])
                    assert set(
                        self.coords[existing_dim]) == old_dim_items_set, f"All items in the mapping dict associated with '{new_dim}' and '{existing_dim}' must be included in .coords['{existing_dim}']"
                    assert len(value[existing_dim][0]) == len(
                        old_dim_items_set), f"There are duplicate items in the mapping dict associated with '{new_dim}' and '{existing_dim}'"
                    if isinstance(value[existing_dim][0], list):
                        kwargs[new_dim][existing_dim][0] = _test_type_and_update(
                            value[existing_dim][0])
                    assert isinstance(
                        kwargs[new_dim][existing_dim][0], np.ndarray), f"Value associated with '{new_dim}' and '{existing_dim}' must be a numpy array. Got {type(kwargs[new_dim][existing_dim][0])}"
                    new_dim_items = value[existing_dim][1]
                    new_dim_items_set = set(new_dim_items)
                    if len(new_dim_items) == len(new_dim_items_set):
                        coords[new_dim] = _test_type_and_update(
                            value[existing_dim][1])
                    else:
                        coords[new_dim] = np.unique(
                            _test_type_and_update(value[existing_dim][1]))
            # this is a list of strings that represent several dimensions named in self.dims. The new dimension is the concatenation of the selected dimensions
            elif isinstance(value, list):
                assert value, "List cannot be empty"
                assert all([isinstance(item, str) for item in value]), "All items in the list must be str"
                assert all([dim in self.dims for dim in value]), "All items in the list must be in dims"
                selected_coords = {dim: self.coords[dim] for dim in value}
                arrays = np.unravel_index(np.arange(self._capacity(
                    selected_coords)), self._shape(selected_coords))
                index = {dim: self.coords[dim][idx] for dim, idx in zip(selected_coords, arrays)}
                coords[new_dim] = _join_str(list(index.values()), sep=":")
                kwargs[new_dim] = {
                    tuple(value): [selected_coords, coords[new_dim]]}
            else:
                raise AssertionError(f"Unexpected type: {type(value)}")
        for dim in self.coords:
            coords[dim] = self.coords[dim]
        long = self.long.insert(**kwargs)
        return Array(data=long, coords=coords)

    def add_dim(self, **kwargs: Union[np.dtype, type, str, int, Dict[str, Union[Dict[str, Any], List[Union[List[str], List[Any]]]]]]) -> 'Array':
        """
        Add new dimensions to the Array.

        Args:
            **kwargs: Keyword arguments specifying the new dimensions and their values.

        Returns:
            A new Array object with the added dimensions.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> new_arr = arr.add_dim(x=0)
            >>> new_arr
            Array(data=array([[[10,  0],
                    [ 0, 20]]]), coords={'x': array([0]), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self.insert(**kwargs)

    def rename(self, **kwargs: str) -> 'Array':
        """
        Rename dimensions of the Array.

        Args:
            **kwargs: Keyword arguments specifying the old dimension names and their new names.

        Returns:
            A new Array object with the renamed dimensions.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> new_arr = arr.rename(dim1='new_dim1')
            >>> new_arr
            Array(data=array([[10.,  0.],
                   [ 0., 20.]]), coords={'new_dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        for olddim, newdim in kwargs.items():
            assert olddim in self.dims, f"Dimension {olddim} must be in dims {self.dims}"
            assert newdim not in self.dims, f"Dimension {newdim} must not be in dims {self.dims}"
        coords = {}
        for dim, elems in self.coords.items():
            if dim in kwargs:
                coords[kwargs[dim]] = elems
            else:
                coords[dim] = elems
        return Array(data=self.data, coords=coords)

    def drop(self, dims: Union[str, List[str]]) -> 'Array':
        """
        Drop specified dimensions from the Array.

        Args:
            dims: A single dimension or a list of dimensions to drop.

        Returns:
            A new Array object with the specified dimensions dropped.

        Example:
            ```python
            >>> coords = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'a'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> new_arr = arr.drop('dim1')
            >>> new_arr
            Array(data=array([10, 20]), coords={'dim2': array([1, 2])})

            ```
        """
        long = self.long.drop(dims=dims)
        coords = {dim: self.coords[dim] for dim in long.dims}
        return Array(data=long, coords=coords)

    def dropna(self) -> 'Array':
        """
        Drop missing values (NaN) from the Array.
        """
        long = self.long
        long = long[long != np.nan]
        return Array(data=long, coords=self.coords)

    def dropinf(self, pos: bool = False, neg: bool = False) -> 'Array':
        """
        Drop infinite values (inf or -inf) from the Array.

        Args:
            pos: Whether to drop positive infinity values.
            neg: Whether to drop negative infinity values.

        Returns:
            A new Array object with infinite values removed.
        """
        assert any([pos, neg]), "pos and neg cannot be both False"
        long = self.long
        if pos:
            long = long[long != np.inf]
        if neg:
            long = long[long != -np.inf]
        return Array(data=long, coords=self.coords)

    def round(self, decimals: int) -> 'Array':
        """
        Round the values in the Array to the specified number of decimal places.

        Args:
            decimals: The number of decimal places to round to.

        Returns:
            A new Array object with the rounded values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long_with_decimals = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10.123, 20.456])
            >>> arr_with_decimals = Array(data=long_with_decimals, coords=coords)
            >>> rounded_arr = arr_with_decimals.round(decimals=1)
            >>> rounded_arr
            Array(data=array([[10.1,  0. ],
                   [ 0. , 20.5]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        ndarray = self.data.round(decimals=decimals)
        coords = self.coords
        return Array(data=ndarray, coords=coords)

    def elems_to_datetime(self, new_dim: str, actual_dim: str, reference_date: str, freq: str, sort_coords: bool = True) -> 'Array':
        """
        Convert elements of a dimension to datetime values and create a new dimension.

        Args:
            new_dim: The name of the new dimension to create.
            actual_dim: The name of the existing dimension to convert.
            reference_date: The reference date to start the datetime range from.
            freq: The frequency of the datetime range.
            sort_coords: Whether to sort the coordinates of the actual dimension.

        Returns:
            A new Array object with the datetime dimension added.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> new_arr = arr.elems_to_datetime(new_dim='date', actual_dim='dim2', reference_date='2022-01-01', freq='D')
            >>> new_arr
            Array(data=array([[[10,  0],
                    [ 0,  0]],
            <BLANKLINE>
                   [[ 0,  0],
                    [ 0, 20]]]), coords={'date': array(['2022-01-01T00:00:00.000000000', '2022-01-02T00:00:00.000000000'],
                  dtype='datetime64[ns]'), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        assert actual_dim in self.dims
        start_date = pd.to_datetime(reference_date)
        t = pd.date_range(start=start_date, periods=self.coords[actual_dim].size, freq=freq)
        if sort_coords:
            new_array = self.insert(**{new_dim: {actual_dim: [np.sort(self.coords[actual_dim]), t]}})
        else:
            new_array = self.insert(**{new_dim: {actual_dim: [self.coords[actual_dim], t]}})
        return new_array

    def elems_to_int(self, new_dim: str, actual_dim: str) -> 'Array':
        """
        Convert elements of a dimension to integer values and create a new dimension.

        Args:
            new_dim: The name of the new dimension to create.
            actual_dim: The name of the existing dimension to convert.

        Returns:
            A new Array object with the integer dimension added.

        Example:
            ```python
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': ['1', '2']}, value=[10, 20])
            >>> arr = Array(data=long, coords={'dim1': ['a', 'b'], 'dim2': ['1', '2']})
            >>> new_arr = arr.elems_to_int(new_dim='int_dim', actual_dim='dim2')
            >>> new_arr
            Array(data=array([[[10,  0],
                    [ 0,  0]],
            <BLANKLINE>
                   [[ 0,  0],
                    [ 0, 20]]]), coords={'int_dim': array([1, 2]), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array(['1', '2'], dtype=object)})

            ```
        """
        serie = pd.Series(data=self.coords[actual_dim])
        serie = serie.str.extract(r"(\d+)", expand=False).astype("int")
        new_array = self.insert(**{new_dim: {actual_dim: [self.coords[actual_dim], serie.values]}})
        return new_array

    def empty(self) -> 'Array':
        """
        Create an empty Array with the same dimensions and coordinates as the original Array.

        Returns:
            A new empty Array object.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> empty_arr = arr.empty()
            >>> empty_arr
            Array(data=array([[0, 0],
                   [0, 0]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return Array(data=({dim: [] for dim in self.dims}, np.array([], dtype=self.data.dtype)), coords=self.coords)

    def choice(self, dim: str, seed: int = 1) -> 'Array':
        """
        Randomly choose elements along a specified dimension based on the Array values as probabilities.

        Args:
            dim: The dimension to perform the choice along.
            seed: The random seed to use for reproducibility.

        Returns:
            A new Array object with the chosen elements.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'a', 'b', 'b'], 'dim2': [1, 2, 1, 2]}, value=[0.3, 0.4, 0.7, 0.6])
            >>> arr = Array(data=long, coords=coords)
            >>> chosen_arr = arr.choice(dim='dim1', seed=42)
            >>> chosen_arr
            Array(data=array([[False, False],
                   [ True,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        rng = np.random.default_rng(seed=seed)
        assert dim in self.dims, f"dim {dim} not in self.dims: {self.dims}"
        axis = self.dims.index(dim)
        probabilities = self.data
        mask = ndarray_choice(p=probabilities, axis=axis, rng=rng)
        assert mask.shape == probabilities.shape
        return Array(data=mask, coords=self.coords)

    def expand(self, **kwargs: Dict[str, Union[np.ndarray, List[str], List[int], List[float], List['np.datetime64'], List['pd.DatetimeIndex'], List['pd.Categorical']]]) -> 'Array':
        """
        Expand the Array with new dimensions and coordinates. It broadcasts the values along the new dimensions.

        Args:
            **kwargs: Keyword arguments specifying the new dimensions and their coordinates.

        Returns:
            A new Array object with the expanded dimensions.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> expanded_arr = arr.expand(new_dim=['x','y'])
            >>> expanded_arr.reorder(reorder=['dim1', 'dim2', 'new_dim'])
            Array(data=array([[[10., 10.],
                    [ 0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.],
                    [20., 20.]]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2]), 'new_dim': array(['x', 'y'], dtype=object)})

            ```
        """
        types = {}
        for dim in kwargs:
            assert dim not in self.dims, f"dim {dim} already exists in self.dims: {self.dims}"
            types[dim] = _test_type_and_update(kwargs[dim]).dtype
        return self.empty().insert(**types).add_elem(**kwargs) + self

    def ufunc(self, dim: str, func: Callable, keepdims: bool = False, **func_kwargs: Any) -> 'Array':
        """
        Apply a universal function along a specified dimension of the Array.

        Args:
            dim: The dimension to apply the function along.
            func: The universal function to apply.
            keepdims: Whether to keep the reduced dimension in the result.
            **func_kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A new Array object with the function applied.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> result_arr = arr.ufunc(dim='dim1', func=np.sum, keepdims=True)
            >>> result_arr
            Array(data=array([[10., 20.],
                   [10., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        axis = self.dims.index(dim)
        result = func(self.todense(), axis=axis,
                      keepdims=keepdims, **func_kwargs)
        if keepdims:
            shape = self.shape
            result = np.broadcast_to(result, shape)
            coords = {d: self.coords[d] for d in self.coords}
        else:
            coords = {d: self.coords[d] for d in self.coords if d != dim}
        return Array(data=result, coords=coords)

    @property
    def df(self) -> Union['pd.DataFrame', 'pl.DataFrame']:
        """
        Get the DataFrame representation of the Array based on the default setting.

        Returns:
            A DataFrame (pandas or polars) representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> arr.df
              dim1  dim2  value
            0    a     1     10
            1    a     2      0
            2    b     1      0
            3    b     2     20

            ```
        """
        if self.df_with == "pandas":
            return self.to_pandas()
        elif self.df_with == "polars":
            return self.to_polars()
        else:
            raise Exception(
                "ka.settings.df_with must be either 'pandas' or 'polars'")

    def todense(self) -> np.ndarray:
        """
        Takes data property and provide a dense numpy array. If data property return a sparse object then convert it to a dense numpy array. If it returns dense object then return the original dense numpy array.

        Returns:
            A dense numpy array.
        """
        if isinstance(self.data, np.ndarray):
            return self.data
        elif _isinstance_optional_pkgs(self.data, 'sp.COO'):
            return self.data.todense()
        else:
            raise Exception(f"Invalid type for 'data': {type(self.data)}")

    def _check_duplicate_indexes(self, indexes: np.ndarray, dims: List[str], coords: Dict[str, np.ndarray]) -> None:
        """
        Check if the indexes contains duplicate values.

        Args:
            indexes: The indexes to check.
            dims: The dimensions of the indexes.
            coords: The coordinates of the indexes.

        Raises:
            ValueError: If the indexes contains duplicate values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
            >>> arr = Array(data=long, coords=coords)
            >>> arr._check_duplicate_indexes(indexes=np.array([0, 0, 1]), dims=['dim1', 'dim2'], coords=coords)
            Traceback (most recent call last):
            ...
            ValueError: The Long object contains duplicate indexes. Duplicate indexes are: [{'loc': {'dim1': 'a', 'dim2': 1}, 'count': 2}].

            ```
        """
        if not np.unique(indexes).size == indexes.size:
            count_repeated = np.bincount(indexes)
            duplicate_indexes = np.where(count_repeated > 1)[0]
            repeated = []
            arrays = np.unravel_index(indexes, self._shape(coords))
            index_dict = {dim: coords[dim][idx]
                          for dim, idx in zip(coords, arrays)}
            for i in duplicate_indexes:
                repeated.append({'loc': {dim: index_dict[dim][np.argmax(
                    indexes == i)] for dim in dims}, 'count': count_repeated[i]})
            raise ValueError(
                f"The Long object contains duplicate indexes. Duplicate indexes are: {repeated}.")


def concat(arrays: List[Array]) -> Array:
    """
    Concatenate multiple Array objects along a new dimension.

    Args:
        arrays: A list of Array objects to concatenate.

    Returns:
        A new Array object with the concatenated arrays.

    Example:
        ```python
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> long1 = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 1]}, value=[10, 20])
        >>> arr1 = Array(data=long1, coords=coords)
        >>> long2 = Long(index={'dim1': ['a', 'b'], 'dim2': [2, 2]}, value=[30, 40])
        >>> arr2 = Array(data=long2, coords=coords)
        >>> concatenated_arr = concat([arr1, arr2])
        >>> concatenated_arr
        Array(data=array([[10, 30],
               [20, 40]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

        ```
    """
    dims = arrays[0].dims[:]
    assert all([isinstance(arr, Array) for arr in arrays]
               ), "All list items must be karray.array"
    assert all([set(arr.dims) == set(dims) for arr in arrays]
               ), "All array must have the same dimensions"
    index = {dim: [] for dim in dims}
    value = []
    [index[dim].append(arr.long.index[dim])
     for arr in arrays for dim in arr.dims]
    index = {dim: np.hstack(index[dim]) for dim in dims}
    [value.append(arr.long.value) for arr in arrays]
    value = np.hstack(value)
    list_of_coords = [arr.coords for arr in arrays]
    coords = union_multi_coords(*list_of_coords)
    return Array(data=(index, value), coords=coords)


def numpy_to_long(array: np.ndarray, dims: List[str]) -> Long:
    """
    Convert a NumPy array to a Long object.

    Args:
        array: The NumPy array to convert.
        dims: The list of dimension names for the Long object.

    Returns:
        A Long object created from the NumPy array.

    Example:
        ```python
        >>> array = np.array([[1, 2, 10], [3, 4, 20]])
        >>> dims = ['dim1', 'dim2']
        >>> long_obj = numpy_to_long(array, dims)
        >>> long_obj
        Long(index={'dim1': array([1, 3]), 'dim2': array([2, 4])}, value=array([10., 20.]))

        ```
    """
    assert isinstance(array, np.ndarray)
    assert isinstance(dims, list)
    assert array.ndim == 2, "Array must be a 2 dimensions array"
    assert len(
        dims) + 1 == len(array.T), f"Numpy array must contain {len(dims) + 1} columns"
    value = array.T[len(dims)]
    _index = {dim: arr for dim, arr in zip(dims, array.T[0:len(dims)])}
    _value = value if issubclass(
        value.dtype.type, float) else value.astype(float)
    return Long(_index, _value)


def _pandas_to_array(df: 'pd.DataFrame', coords: Union[Dict[str, np.ndarray], None] = None) -> Dict[str, Union[Tuple[Dict[str, np.ndarray], np.ndarray], Dict[str, np.ndarray]]]:
    """
    Convert a pandas DataFrame to an Array dictionary.

    Args:
        df: The pandas DataFrame to convert.
        coords: The coordinates for the Array.

    Returns:
        A dictionary representing the Array data and coordinates.

    Example:
        ```python
        >>> df = pd.DataFrame({'dim1': ['a', 'b'], 'dim2': [1, 2], 'value': [10, 20]})
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> array_dict = _pandas_to_array(df, coords)
        >>> array_dict
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10, 20])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

        ```
    """
    assert "value" in df.columns, "Column named 'value' must exist"
    value = df["value"].values
    df = df.drop(labels="value", axis=1)
    index = {}
    for col in df.columns:
        index[col] = df[col].values
    return dict(data=(index, value), coords=coords)


def from_pandas(df: 'pd.DataFrame', coords: Union[Dict[str, np.ndarray], None] = None) -> Array:
    """
    Create an Array object from a pandas DataFrame.

    Args:
        df: The pandas DataFrame to convert.
        coords: The coordinates for the Array.

    Returns:
        An Array object created from the pandas DataFrame.

    Example:
        ```python
        >>> df = pd.DataFrame({'dim1': ['a', 'b'], 'dim2': [1, 2], 'value': [10, 20]})
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> arr = from_pandas(df, coords)
        >>> arr
        Array(data=array([[10,  0],
               [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

        ```
    """
    return Array(**_pandas_to_array(df, coords=coords))


def _polars_to_array(df: 'pl.DataFrame', coords: Union[Dict[str, np.ndarray], None] = None) -> Dict[str, Union[Tuple[Dict[str, np.ndarray], np.ndarray], Dict[str, np.ndarray]]]:
    """
    Convert a polars DataFrame to an Array dictionary.

    Args:
        df: The polars DataFrame to convert.
        coords: The coordinates for the Array.

    Returns:
        A dictionary representing the Array data and coordinates.

    Example:
        ```python
        >>> df = pl.DataFrame({'dim1': ['a', 'b'], 'dim2': [1, 2], 'value': [10, 20]})
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> array_dict = _polars_to_array(df, coords)
        >>> array_dict
        {'data': ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, array([10, 20])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

        ```
    """
    assert "value" in df.columns, "Column named 'value' must exist"
    value = df["value"].to_numpy()
    df = df.drop("value")
    index = df.to_dict(as_series=False)
    return dict(data=(index, value), coords=coords)


def from_polars(df: 'pl.DataFrame', coords: Union[Dict[str, np.ndarray], None] = None) -> Array:
    """
    Create an Array object from a polars DataFrame.

    Args:
        df: The polars DataFrame to convert.
        coords: The coordinates for the Array.

    Returns:
        An Array object created from the polars DataFrame.

    Example:
        ```python
        >>> df = pl.DataFrame({'dim1': ['a', 'b'], 'dim2': [1, 2], 'value': [10, 20]})
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> arr = from_polars(df, coords)
        >>> arr
        Array(data=array([[10,  0],
               [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

        ```
    """
    return Array(**_polars_to_array(df, coords=coords))


def from_feather_to_dict(path: str, use_threads: bool = True, with_: Union[str, None] = None) -> Dict[str, Union[Tuple[Dict[str, np.ndarray], np.ndarray], Dict[str, np.ndarray]]]:
    """
    Load an Array dictionary from a Feather file.

    Args:
        path: The path to the Feather file.
        use_threads: Whether to use threads when reading the Feather file.
        with_: The library to use for loading the Feather file ('pandas' or 'polars').

    Returns:
        A dictionary representing the Array data and coordinates loaded from the Feather file.

    Example:
        ```python
        >>> array_dict = from_feather_to_dict('tests/data/array.feather', with_='pandas')
        >>> array_dict
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10, 20])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

        ```
    """
    assert with_ in ["pandas", "polars"]
    restored_table = feather.read_table(
        path, use_threads=use_threads, memory_map=True)
    column_names = restored_table.column_names
    assert "value" in column_names, "Column named 'value' must exist"
    custom_meta_key = 'karray'
    if custom_meta_key.encode() in restored_table.schema.metadata:
        restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode(
        )]
        restored_meta = json.loads(restored_meta_json)
        assert "coords" in restored_meta
        if with_ == "pandas":
            return _pandas_to_array(df=restored_table.to_pandas(), coords=restored_meta['coords'])
        elif with_ == "polars":
            return _polars_to_array(df=pl.from_arrow(restored_table), coords=restored_meta['coords'])
    else:
        if with_ == "pandas":
            return _pandas_to_array(df=restored_table.to_pandas(split_blocks=True, self_destruct=True), coords=None)
        elif with_ == "polars":
            return _polars_to_array(df=pl.from_arrow(restored_table), coords=None)


def from_feather(path: str, use_threads: bool = True, with_: str = 'pandas') -> Array:
    """
    Load an Array object from a Feather file.

    Args:
        path: The path to the Feather file.
        use_threads: Whether to use threads when reading the Feather file.
        with_: The library to use for loading the Feather file ('pandas' or 'polars').

    Returns:
        An Array object loaded from the Feather file.

    Example:
        ```python
        >>> arr = from_feather('tests/data/array.feather', with_='pandas')
        >>> arr
        Array(data=array([[10,  0],
               [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

        ```
    """
    return Array(**from_feather_to_dict(path=path, use_threads=use_threads, with_=with_))


def _csv_to_array(path: str, coords: Union[Dict[str, np.ndarray], None] = None, delimiter: str = ',', encoding: str = 'utf-8') -> Dict[str, Union[Tuple[Dict[str, np.ndarray], np.ndarray], Dict[str, np.ndarray]]]:
    """
    Convert a CSV file to an Array dictionary.

    Args:
        path: The path to the CSV file.
        coords: The coordinates for the Array.
        delimiter: The delimiter used in the CSV file.
        encoding: The encoding of the CSV file.

    Returns:
        A dictionary representing the Array data and coordinates.

    Example:
        ```python
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> array_dict = _csv_to_array('tests/data/array.csv', coords=coords)
        >>> array_dict
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10, 20])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

        ```
    """
    with open(file=path, mode='r', encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        index = {}
        headings = next(reader)
        assert "value" in headings, "Column named 'value' must exist"
        for i, col in enumerate(headings):
            first_value_str = headings[col]
            str_strip = first_value_str.lstrip('-')
            if '.' in str_strip:
                try:
                    float(str_strip)
                    dtype = float
                except ValueError:
                    dtype = np.object_
            else:
                try:
                    int(str_strip)
                    dtype = int
                except ValueError:
                    dtype = np.object_
            if col == 'value':
                assert dtype == float or dtype == int, f"Column named 'value' must be of type int or float. Got {dtype}"
                value = np.loadtxt(
                    path, skiprows=1, delimiter=delimiter, usecols=i, dtype=dtype)
            else:
                index[col] = np.loadtxt(
                    path, skiprows=1, delimiter=delimiter, usecols=i, dtype=dtype)
    return dict(data=(index, value), coords=coords)


def from_csv_to_dict(path: str, coords: Union[Dict[str, np.ndarray], None] = None, delimiter: str = ',', encoding: str = 'utf-8', with_: str = 'csv') -> Dict[str, Union[Tuple[Dict[str, np.ndarray], np.ndarray], Dict[str, np.ndarray]]]:
    """
    Load an Array dictionary from a CSV file.

    Args:
        path: The path to the CSV file.
        coords: The coordinates for the Array.
        delimiter: The delimiter used in the CSV file.
        with_: The library to use for loading the CSV file ('csv', 'pandas', or 'polars').

    Returns:
        A dictionary representing the Array data and coordinates loaded from the CSV file.

    Example:
        ```python
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> array_dict = from_csv_to_dict('tests/data/array.csv', coords=coords, with_='csv')
        >>> array_dict
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10, 20])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

        ```
    """
    assert with_ in ["csv", "pandas", "polars"]
    if with_ == "csv":
        return _csv_to_array(path=path, coords=coords, delimiter=delimiter, encoding=encoding)
    if with_ == "pandas":
        df = pd.read_csv(path)
        return _pandas_to_array(df=df, coords=coords)
    elif with_ == "polars":
        df = pl.read_csv(path)
        return _polars_to_array(df=df, coords=coords)


def from_csv(path: str, coords: Union[Dict[str, np.ndarray], None] = None, delimiter: str = ',', encoding: str = 'utf-8', with_: str = 'csv') -> Array:
    """
    Load an Array object from a CSV file.

    Args:
        path: The path to the CSV file.
        coords: The coordinates for the Array.
        delimiter: The delimiter used in the CSV file.
        encoding: The encoding of the CSV file.
        with_: The library to use for loading the CSV file ('csv', 'pandas', or 'polars').

    Returns:
        An Array object loaded from the CSV file.

    Example:
        ```python
        >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        >>> arr = from_csv('tests/data/array.csv', coords=coords, with_='csv')
        >>> arr
        Array(data=array([[10,  0],
               [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

        ```
    """
    return Array(**from_csv_to_dict(path, coords=coords, delimiter=delimiter, encoding=encoding, with_=with_))


def _join_str(arr: List[np.ndarray], sep: str) -> np.ndarray:
    """
    Join a list of string arrays into a single string array using a separator.

    Args:
        arr: A list of string arrays to join.
        sep: The separator to use for joining the strings.

    Returns:
        A single string array with the joined strings.

    Example:
        ```python
        >>> arr = [np.array(['a', 'b']), np.array(['1', '2'])]
        >>> sep = ':'
        >>> joined_arr = _join_str(arr, sep)
        >>> joined_arr
        array(['a:1', 'b:2'], dtype='<U3')

        ```
    """
    rows = arr[0].shape[0]
    columns = len(arr)
    separator_str = np.repeat([sep], rows)

    arrays = []
    for i in range(columns):
        arrays.append(arr[i].astype(str))
        if i != columns-1:
            arrays.append(separator_str)
    return functools_reduce(lambda x, y: np.char.add(x, y), arrays)


def ndarray_choice(p: Union[np.ndarray, 'sp.COO'], axis: int, rng: np.random.Generator) -> Union[np.ndarray, 'sp.COO']:
    """
    Randomly choose elements along a specified axis based on the given probabilities.

    Args:
        p: The array of probabilities.
        axis: The axis along which to perform the choice.
        rng: The random number generator to use.

    Returns:
        An array of boolean values indicating the chosen elements.

    Example:
        ```python
        >>> p = np.array([[0.1, 0.9], [0.7, 0.3]])
        >>> axis = 1
        >>> rng = np.random.default_rng(seed=42)
        >>> chosen_arr = ndarray_choice(p, axis, rng)
        >>> chosen_arr
        array([[False,  True],
               [ True, False]])

        ```
    """
    def _masking(p, axis):
        shape = [nr for i, nr in enumerate(p.shape) if i != axis]
        rand = rng.random(tuple(shape))
        r = np.expand_dims(rand, axis=axis)

        if isinstance(p, np.ndarray):
            cum = np.cumsum(p, axis=axis)
        elif _isinstance_optional_pkgs(p, 'sp.COO'):
            cum = np.cumsum(p.todense(), axis=axis)
        else:
            raise Exception(
                "sparse package must be installed to use this function with settings.data_type = 'sparse'. Otherwise, set settings.data_type = 'dense'")

        assert np.allclose(cum.max(axis=axis),
                           1.0), "Probabilities do not sum to 1"
        mask = (cum > r)
        return mask

    def _unravel(mask, axis):
        args = mask.argmax(axis=axis, keepdims=True)
        idx = np.unravel_index(np.arange(args.size), args.shape)
        args_ravel = args.ravel()
        new_idx = [arr if i != axis else args_ravel for i,
                   arr in enumerate(idx)]
        return new_idx

    def _nd_bool(idxs, shape, size):
        indexes = np.ravel_multi_index(idxs, shape)
        flatten_dense = np.empty((size,), dtype=bool)
        flatten_dense[:] = False
        flatten_dense[indexes] = True
        nd_dense = flatten_dense.view().reshape(shape)
        return nd_dense

    shape = p.shape
    size = p.size
    mask = _masking(p, axis)
    idxs = _unravel(mask, axis)
    del mask
    return _nd_bool(idxs, shape, size)


def union_multi_coords(*args: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Union multiple coordinate dictionaries.

    Args:
        *args: Variable length argument list of coordinate dictionaries.

    Returns:
        A dictionary with the union of coordinates from all input dictionaries.

    Example:
        ```python
        >>> coords1 = {'dim1': np.array([1, 2]), 'dim2': np.array([3, 4])}
        >>> coords2 = {'dim1': np.array([1, 2]), 'dim2': np.array([5, 6])}
        >>> union_multi_coords(coords1, coords2)
        {'dim1': array([1, 2]), 'dim2': array([3, 4, 5, 6])}

        ```
    """
    assert all([tuple(coords) == tuple(args[0]) for coords in args])
    dims = list(args[0])
    new_coords = {}
    for coords in args:
        for dim in dims:
            if dim not in new_coords:
                new_coords[dim] = coords[dim]
            else:
                if new_coords[dim].size == coords[dim].size:
                    if all(new_coords[dim] == coords[dim]):
                        continue
                    else:
                        new_coords[dim] = np.union1d(
                            new_coords[dim], coords[dim])
                elif set(new_coords[dim]).issubset(set(coords[dim])):
                    new_coords[dim] = coords[dim]
                elif set(coords[dim]).issubset(set(new_coords[dim])):
                    continue
                else:
                    new_coords[dim] = np.union1d(new_coords[dim], coords[dim])
    return new_coords


def _test_type_and_update(item: Union[List[str], List[int], List[float], List[np.datetime64], np.ndarray]) -> np.ndarray:
    """
    Test the type of the input item and update it accordingly.

    Args:
        item: Input item as a list of strings, integers, floats, datetime64, or a numpy array.

    Returns:
        An updated numpy array based on the input item type.

    Example:
        ```python
        >>> _test_type_and_update([1, 2, 3])
        array([1, 2, 3])
        >>> _test_type_and_update(['a', 'b', 'c'])
        array(['a', 'b', 'c'], dtype=object)

        ```
    """
    if len(item) == 0:
        if isinstance(item, np.ndarray):
            return item
        else:
            return np.array([])
    else:
        if isinstance(item, np.ndarray):
            if issubclass(type(item[0]), str):
                if issubclass(item.dtype.type, np.object_):
                    variable_out = item
                elif issubclass(item.dtype.type, str):
                    variable_out = item.astype('object')
                else:
                    raise Exception(f"Type: {type(item[0])} not implemented. Item: {item}")
            elif issubclass(item.dtype.type, np.object_):
                if issubclass(type(item[0]), int):
                    variable_out = item.astype(np.int32)
                elif isinstance(type(item[0]), float):
                    variable_out = item.astype(np.float32)
                else:
                    raise Exception(f"Type: {type(item[0])} not implemented. Item: {item}")
            elif issubclass(item.dtype.type, (np.int16, np.int32, np.int64)):
                variable_out = item
            elif issubclass(item.dtype.type, (np.float16, np.float32, np.float64)):
                variable_out = item
            elif issubclass(item.dtype.type, np.datetime64):
                variable_out = item
            else:
                raise Exception(f"Type: {type(item[0])} not implemented. Item: {item}")
        elif isinstance(item, list):
            value_type = type(item[0])
            if issubclass(value_type, str):
                selected_type = 'object'
            elif issubclass(value_type, int):
                selected_type = np.int32
            elif issubclass(value_type, float):
                selected_type = np.float32
            elif issubclass(value_type, np.datetime64):
                selected_type = 'datetime64[ns]'
            elif issubclass(value_type, (np.int16, np.int32, np.int64)):
                selected_type = item[0].dtype
            elif issubclass(value_type, (np.float16, np.float32, np.float64)):
                selected_type = item[0].dtype
            else:
                raise Exception(
                    f"Type: {type(item[0])} not implemented. Item: {item}")
            variable_out = np.array(item, dtype=selected_type)
        elif _isinstance_optional_pkgs(item, 'pd.DatetimeIndex'):
            variable_out = item.values
        elif _isinstance_optional_pkgs(item, 'pd.Categorical'):
            variable_out = item.to_numpy(copy=True)
        else:
            Exception(f"Type: {type(item)} not implemented. Item: {item}")

        assert isinstance(variable_out, np.ndarray), f"Type: {type(item)} not implemented. Item: {item}"
        assert variable_out.ndim == 1, "Only 1D arrays are supported"
        return variable_out


def _test_type_and_update_value(value: Union[np.ndarray, list, float, int, bool, np.bool_, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]) -> np.ndarray:
    assert isinstance(value, (np.ndarray, list, float, int, bool, np.bool_, np.int16, np.int32, np.int64, np.float16,
                      np.float32, np.float64)), f"Value attribute must be a numpy array, list, float, int, or bool. Got {type(value)}"
    if isinstance(value, np.ndarray):
        assert issubclass(value.dtype.type, (np.bool_, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
                          ), "dtype suppoerted for value attribute: np.bool_, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64"
        if value.ndim == 0:
            value = value.reshape((value.size,))
    elif isinstance(value, (float, int, bool, np.bool_, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
        value = np.array([value])
    elif isinstance(value, list):
        if value:
            if isinstance(value[0], str):
                raise NotImplementedError(
                    "Long object does not support string values for value attribute")
            elif isinstance(value[0], (int, float, bool)):
                value = np.array(value)
            elif isinstance(value[0], np.ndarray):
                if issubclass(value[0].dtype.type, (np.bool_, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
                    value = np.array(value, dtype=value[0].dtype)
                else:
                    Exception("Nested arrays not supported")
        else:
            value = np.array(value, dtype=None)
    else:
        Exception(
            f"Value attribute must be a numpy array, list, float, int, or bool. Got {type(value)}")
    assert isinstance(
        value, np.ndarray), f"After validation, Value attribute must be a numpy array. Got {type(value)}"
    assert value.ndim == 1, "Only 1D arrays are supported"
    return value


def _format_bytes(size: int) -> str:
    """
    Format a byte size as a human-readable string.

    Args:
        size: Size in bytes.

    Returns:
        A human-readable string representation of the byte size.

    Example:
        ```python
        >>> _format_bytes(1024)
        '1.0 KB'
        >>> _format_bytes(1048576)
        '1.0 MB'

        ```
    """
    power_labels = {40: "TB", 30: "GB", 20: "MB", 10: "KB"}
    for power, label in power_labels.items():
        if size >= 2 ** power:
            approx_size = size / 2 ** power
            return f"{approx_size:.1f} {label}"
    return f"{size} bytes"


def _is_in_range(start: int, end: int, lower: int, upper: int, step: int = 1) -> bool:
    """
    Check before passing slices if they are in the expected range.

    Args:
        start: Start of the slice
        end: End of the slice
        upper: Upper bound
        lower: Lower bound
        step: Step size

    Raises:
        AssertionError: If the slice is not in the expected range.

    Returns:
        True if the slice is in the expected range, raises an exception otherwise.
    """
    assert step in [1, -1]
    if start > 0:
        assert start < upper and start >= lower, f"{start=} < {upper=} and {start=} >= {lower=}"
        assert end > start or end < 0, f"{end=} > {start=} or {end=} < 0"
        if end < 0:
            assert end >= -upper and end <= -lower, f"{end=} >= {-upper=} and {end=} <= {-lower=}"
        else:
            assert end <= upper and end >= lower, f"{end=} <= {upper=} and {end=} >= {lower=}"
    elif start < 0:
        assert start > -upper and start <= -lower, f"{start=} > {-upper=} and {start=} <= {-lower=}"
        assert end > -upper and end <= -lower, f"{end=} > {-upper=} and {end=} <= {-lower=} as {start=} < 0"
        assert end > start and end < 0, f"{end=} > {start=} and {end=} < 0 as {start=} < 0"
    else:
        assert start < upper and start >= lower, f"{start=} < {upper=} and {start=} >= {lower=}"
        assert end > start or end < 0, f"{end=} > {start=} or {end=} < 0 as {start=} < 0"
        if end < 0:
            assert end >= -upper and end < -lower, f"{end=} >= {-upper=} and {end=} < {-lower=}"
        else:
            assert end <= upper and end > lower, f"{end=} <= {upper=} and {end=} > {lower=}"
    return True
