 
import csv
import json
import numpy as np
from html import escape
from importlib.util import find_spec
from functools import reduce as functools_reduce
from typing import Any, Dict, Iterator, List, Tuple, Union, Callable, Optional, Literal
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



class Settings:
    def __init__(self) -> None:
        """
        Initialize the Settings object with default values.

        The Settings object holds various configuration options for the Array object.
        It includes settings for display, data types, and other behavior.

        """
        self.order = None
        self.rows_display = slice(0, 16)
        self.decimals_display = 2
        self.fill_value = False
        self.data_obj = "dense"  # 'dense' or 'sparse'
        self.dense_dtype = None
        self.sparse_dtype = None
        self.allow_broadcast = False
        self.engine = "pandas"  # 'pandas' or 'polars'
        self.dense_dataframe = False


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
    all_types_as_string = ['sp.COO', 'pd.DatetimeIndex', 'pd.Categorical', 'pd.DataFrame', 'pl.DataFrame', 'pa.Table']
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
                    types_list = [pd.DatetimeIndex, pd.Categorical, pd.DataFrame]

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


class Array:
    def __init__(self, data: Union[Tuple[Dict[str, Union[np.ndarray, List[str], List[int], List[float]]], Union[np.ndarray, List[float], List[int]]], np.ndarray, 'sp.COO'], coords: Union[Dict[str, Union[np.ndarray, List[str], List[int], List[float], List[np.datetime64]]], None] = None) -> None:
        """
        Initialize an Array object.

        Args:
            data: The data for the Array object. It can be a dense numpy array, a sparse COO array, or a tuple containing an index dictionary and a value array.
            coords: A dictionary representing the coordinates of the Array object.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr
            Array(data=array([[10,  0],
                   [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        self.__dict__["_repo"] = {}
        self.dense = None
        self.sparse = None
        self.coords = None
        self.allow_broadcast = settings.allow_broadcast
        self.data_obj = settings.data_obj
        self.fill_value = settings.fill_value
        self.order = settings.order
        self.dense_dtype = settings.dense_dtype
        self.sparse_dtype = settings.sparse_dtype
        self.engine = settings.engine
        self.dense_dataframe = settings.dense_dataframe
        self.rows_display = settings.rows_display
        self.decimals_display = settings.decimals_display
        self._attr_constructor(**self._check_input(data, coords))
        return None

    def _check_input(self, data: Union[np.ndarray, 'sp.COO'], coords: Union[Dict[str, Union[np.ndarray, List[str], List[int], List[float], List[np.datetime64]]]]) -> Tuple[Union[np.ndarray, None], Union['sp.COO',None], dict]:
        """
        Check the input for the Array object.

        Args:
            data: The data for the Array object. It can be a dense numpy array, a sparse COO array, or a tuple containing an index dictionary and a value array.
            coords: A dictionary representing the coordinates of the Array object.

        Returns:
            A tuple containing the dense array, sparse array, index dictionary, value array, and coordinates.

        Raises:
            AssertionError: If the input data or coordinates are invalid.
        """
        if isinstance(data, (int, float, bool, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            assert isinstance(coords, dict), "coords must be a dictionary"
            dense = data
            sparse = None
            index = None
            value = None
        elif _isinstance_optional_pkgs(data, 'sp.COO'):
            assert isinstance(coords, dict), "coords must be a dictionary"
            dense = None
            sparse = data
            index = None
            value = None
        elif isinstance(data, tuple):
            assert len(data) == 2
            assert isinstance(coords, (dict, type(None))), "coords must be a dictionary"
            assert isinstance(data[0], dict), "data[0] must be a dictionary"
            assert isinstance(data[1], (np.ndarray, list, int, float, bool, np.int_, np.uint, np.float_, np.bool_)), "data[0] must be a numpy array, list, bool, or any number type"
            dense = None
            sparse = None
            index = data[0]
            value = _test_type_and_update_value(data[1])
        else:
            raise AssertionError("Invalid input for 'data'. Must be a numpy array or a sparse COO array. Make sure sparse is installed")

        if coords is not None:
            for dim in coords:
                assert isinstance(dim, str), "coords must be a dictionary with string keys"
                coords[dim] = _test_type_and_update(coords[dim])
                assert coords[dim].ndim == 1
                assert coords[dim].size == np.unique(
                    coords[dim]).size, f"coords elements of dim '{dim}' must be unique. {coords[dim].size=}, {np.unique(coords[dim]).size=}"
        
        if dense is not None:
            assert dense.ndim == len(coords)
            assert dense.shape == tuple(self._shape(coords))
            assert dense.size == self._capacity(coords)
        elif sparse is not None:
            assert sparse.ndim == len(coords)
            assert sparse.shape == tuple(self._shape(coords))
            assert sparse.nnz <= self._capacity(coords)
        else:
            index, coords = self._index_coords_prepare(index, coords)
        return dict(index=index, value=value, sparse=sparse, dense=dense, coords=coords)
    
    @staticmethod
    def _index_coords_prepare(index: Dict[str, np.ndarray], coords: Union[Dict[str, np.ndarray],None]=None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        for dim in index:
            assert isinstance(dim, str), "index must be a dictionary with string keys"
            index[dim] = _test_type_and_update(index[dim])
            if coords is not None:
                assert set(index[dim]).issubset(set(coords[dim])), f"index elements for dimension '{dim}' must be a subset of coords"
        if coords is None:
            coords = {k: np.unique(v) for k, v in index.items()}
        return index, coords

    def _attr_constructor(self, index: Union[Dict[str, np.ndarray], None], value: Union[np.ndarray, None], sparse: Union['sp.COO',None], dense: Union[np.ndarray, None], coords: Dict[str, np.ndarray]) -> None:
        """
        Set the attributes of the Array object based on the provided data and coordinates.

        Args:
            index: The index dictionary.
            value: The value array.
            sparse: The sparse COO array.
            dense: The dense array.
            coords: The coordinates of the array.

        Returns:
            None
        """
        assert coords is not None, "coords cannot be None"
        if dense is not None:
            order = self._order_with_preference(list(coords), self.order)
            self.dense, self.coords = self._reorder(dense, coords, order).values()
        elif sparse is not None:
            order = self._order_with_preference(list(coords), self.order)
            self.sparse, self.coords = self._reorder(sparse, coords, order).values()
        else:
            # index and value
            filler, dtype = self._filler_and_dtype(value, self.fill_value)
            order = self._order_with_preference(list(coords), self.order)
            coords = {dim: coords[dim] for dim in order}
            self.coords = coords
            if self.data_obj == "sparse":
                self.sparse = self._from_index_value(index, value, coords, self._shape(coords), dtype, filler, self.data_obj)
            elif self.data_obj == "dense":
                self.dense = self._from_index_value(index, value, coords, self._shape(coords), dtype, filler, self.data_obj)

        return None
    
    @staticmethod
    def _from_index_value(index: Dict[str, np.ndarray], value: np.ndarray, coords: Dict[str, np.ndarray], shape: Tuple[int, ...], dtype: Union[np.int_, np.float_, np.bool_], fill_value: Union[int, float, bool], data_obj: str) -> Union[sp.COO, np.ndarray]:
        """
        Create a sparse COO array or dense array from the index and value.

        Args:
            index: The index of the array.
            value: The value of the array.
            coords: The coordinates of the array.
            dtype: The data type of the array.
            fill_value: The fill value of the array.
            data_obj: The data object of the array.

        Returns:
            A sparse COO array or dense array.
        """
        indices = tuple([np.argsort(coords[dim])[np.searchsorted(coords[dim], index[dim], sorter=np.argsort(coords[dim]))] for dim in coords])
        if data_obj == "dense":
            dense = np.full(shape=shape, fill_value=fill_value, dtype=dtype)
            dense[indices] = value.astype(dtype)
            return dense
        elif data_obj == "sparse":
            return sp.COO(coords=indices, data=value.astype(dtype), shape=shape, fill_value=fill_value)

    def __repr__(self) -> str:
        """
        Return a string representation of the Array object.

        Returns:
            A string representation of the Array object.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> repr(arr)
            "Array(data=array([[10,  0],
                   [ 0, 20]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})"

            ```
        """
        return f"Array(data={repr(self.dense)}, coords={self.coords})"

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the Array object.

        Returns:
            An HTML string representation of the Array object.
        """
        def generate_table_rows(data):
            rows = []
            ndim = data['value'].ndim
            if ndim == 0:
                rows.append(f'<tr><td>{data["value"]}</td></tr>')
                return '\n'.join(rows)

            else:
                for i in range(len(data['value'])):
                    row = ''.join(f'<td>{data[dim][i]}</td>' for dim in self.dims)
                    row += f'<td>{data["value"][i]}</td>'
                    rows.append(f'<tr>{row}</tr>')
                return '\n'.join(rows)

        def generate_coordinates_table():
            rows = []
            for dim in self.coords:
                row = f'<tr><td>{dim}</td><td>{len(self.coords[dim])}</td><td>{self.coords[dim].dtype}</td><td>{escape(str(self.coords[dim]))}</td></tr>'
                rows.append(row)
            return '\n'.join(rows)

        def format_bytes(size: int) -> str:
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
        
        # Prepare the data
        data_dict = self.to_dict(rows=self.rows_display, dense=self.dense_dataframe)
        headers = ''.join(f'<th><b>{header}</b></th>' for header in data_dict)
        ndim = data_dict["value"].ndim

        if ndim == 0:
            length = 1
        else:
            length = len(data_dict["value"])

        # Generate the HTML
        html = f'''
            <div>
                <details open>
                    <summary><strong>Array</strong></summary>
                    <table>
                        <tr><th><b>Attribute</b></th><th><b>Value</b></th></tr>
                        <tr><td>Data Object format</td><td>{self.data_obj}</td></tr>
                        <tr><td>Data Object Type</td><td>{self.data.dtype}</td></tr>
                        <tr><td>Data Object Size</td><td>{format_bytes(self.dense.nbytes)}</td></tr>
                        <tr><td>Dimensions</td><td>{self.dims}</td></tr>
                        <tr><td>Shape</td><td>{self.shape}</td></tr>
                        <tr><td>Capacity</td><td>{self.capacity}</td></tr>
                    </table>
                </details>
                <details>
                    <summary><strong>Coordinates</strong></summary>
                    <table>
                        <tr><th>Dimension</th><th>Length</th><th>Type</th><th>Items</th></tr>
                        {generate_coordinates_table()}
                    </table>
                </details>
                <details>
                    <summary><strong>Data</strong></summary>
                    <table>
                        <tr>{headers}</tr>
                        {generate_table_rows(data_dict)}
                        {'<tr><td colspan="3">...</td></tr>' if length > self.rows_display.stop else ''}
                    </table>
                </details>
            </div>
        '''
        return f'<div>{css_style}{html}</div>'

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
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.dense_dtype = 'float32'

            ```
        """
        if name == "dense":
            if value is not None and self.dense_dtype is not None:
                if issubclass(value.dtype.type, (np.float16, np.float32, np.float64)):
                    assert self.dense_dtype in ["float16", "float32", "float64"]
                    value = value.astype(self.dense_dtype)
        elif name == "sparse":
            if value is not None and self.sparse_dtype is not None:
                if issubclass(value.dtype.type, (np.float16, np.float32, np.float64)):
                    assert self.sparse_dtype in ["float16", "float32", "float64"]
                    value = value.astype(self.sparse_dtype)
        elif name == "dense_dtype":
            if value is not None:
                assert value in ["float16", "float32", "float64"]
                self._repo['dense'] = self._repo['dense'].astype(value) if self._repo['dense'] is not None else None
        elif name == "sparse_dtype":
            if value is not None:
                assert value in ["float16", "float32", "float64"]
                self._repo['sparse'] = self._repo['sparse'].astype(value) if self._repo['sparse'] is not None else None
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
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.dense
            array([[10,  0],
                   [ 0, 20]])

            ```
        """
        if name.startswith('_'):
            raise AttributeError(name)
        elif name == 'dense':
            if name in self._repo:
                if self._repo[name] is None:
                    if self._repo["sparse"] is not None:
                        self._repo[name] = self._sparse_to_dense(self.sparse, self.coords)
                    else:
                        raise Exception("No data available")
                    return self._repo[name]
                else:
                    return self._repo[name]
            else:
                if self._repo["sparse"] is not None:
                    self._repo[name] = self._sparse_to_dense(self.sparse, self.coords)
                else:
                    raise Exception("No data available")
                return self._repo[name]
        elif name == 'sparse':
            if name in self._repo:
                if self._repo[name] is None:
                    if self._repo["dense"] is not None:
                        self._repo[name] = self._dense_to_sparse(self.dense, self.coords)
                    else:
                        raise Exception("No data available")
                    return self._repo[name]
                else:
                    return self._repo[name]
            else:
                if self._repo["dense"] is not None:
                    self._repo[name] = self._dense_to_sparse(self.dense, self.coords)
                else:
                    raise Exception("No data available")
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
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.data
            array([[10.,  0.],
                   [ 0., 20.]])

            ```
        """
        if self.data_obj == 'sparse':
            return self.sparse
        elif self.data_obj == 'dense':
            return self.dense
        else:
            raise ValueError(
                f"data_obj must be 'sparse' or 'dense', not {self.data_obj}")

    @staticmethod
    def _shape(coords):
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
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.shape
            [2, 2]

            ```
        """
        return self._shape(self.coords)

    @staticmethod
    def _capacity(coords):
        """
        Calculate the capacity of the Array based on the provided coordinates.

        Args:
            coords: The coordinates of the Array.

        Returns:
            The total number of elements the Array can hold.
        """
        return int(np.prod(Array._shape(coords)))

    @property
    def capacity(self) -> int:
        """
        Get the capacity of the Array.

        Returns:
            The total number of elements the Array can hold.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.dims
            ['dim1', 'dim2']

            ```
        """
        return list(self.coords)

    def dindex(self, rows: Optional[slice] = None) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Get the dense index of the Array.

        Returns:
            An iterator yielding tuples of dimension names and corresponding dense index arrays.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> for dim, idx in arr.dindex():
            ...     print(dim, idx)
            dim1 ['a' 'a' 'b' 'b']
            dim2 [1 2 1 2]
            value [10.  0.  0. 20.]

            ```
        """
        if len(self.coords) == 0:
            yield ('value', self.dense)
        else:
            if rows is None:
                arrays = np.unravel_index(np.arange(self.capacity), self.shape)
            else:
                arrays = np.unravel_index(np.arange(rows.start or 0, rows.stop, rows.step or 1), self.shape)
            for dim, idx in zip(self.coords, arrays):
                yield (dim, self.coords[dim][idx])
            yield ('value', self.dense[arrays])

    def sindex(self, rows: Optional[slice] = None) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Get the sparse index of the Array.

        Returns:
            An iterator yielding tuples of dimension names and corresponding sparse index arrays.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> for dim, idx in arr.sindex():
            ...     print(dim, idx)
            dim1 ['a' 'b']
            dim2 [1 2]
            value [10. 20.]

            ```
        """
        if len(self.coords) == 0:
            yield ('value', np.array(self.sparse.data[0], dtype=self.sparse.data.dtype))
        else:
            if rows is None:
                arrays = self.sparse.coords
            else:
                positions = np.ravel_multi_index(self.sparse.coords, self.shape)
                test_rng = np.arange(rows.start or 0, rows.stop, rows.step or 1)
                mask = np.isin(positions, test_rng)
                rng = positions[mask]
                arrays = np.unravel_index(rng, self.shape)
            for dim, idx in zip(self.coords, arrays):
                yield (dim, self.coords[dim][idx])
            if rows is None:
                yield ('value', self.sparse.data)
            else:
                yield ('value', self.sparse[arrays])
                

    def _filler_and_dtype(self, dense: np.ndarray, fill_missing: Union[float, int, bool, None]) -> Tuple[Union[float, int, bool], np.dtype]:
        """
        Determine the filler value and data type based on the dense array and fill_missing value.

        Args:
            dense: The dense array.
            fill_missing: The value to use for missing elements.

        Returns:
            A tuple containing the filler value and the data type.

        Example:
            ```python
            >>> dense = np.array([[10.0, 0.0], [0.0, 20.0]])
            >>> arr = Array(data=dense)
            >>> arr._filler_and_dtype(dense, fill_missing=0.0)
            (0.0, dtype('float64'))

            ```
        """
        if issubclass(dense.dtype.type, (np.float16, np.float32, np.float64)):
            dtype = dense.dtype
            if np.isnan(fill_missing) or np.isinf(fill_missing):
                filler = fill_missing
            elif isinstance(fill_missing, float):
                filler = fill_missing
            elif isinstance(fill_missing, (int, bool)):
                filler = float(fill_missing)
            else:
                raise TypeError("fill_missing must be a float, int or bool")
        elif issubclass(dense.dtype.type, (np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64)):
            dtype = dense.dtype
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
        elif issubclass(dense.dtype.type, np.bool_):
            dtype = dense.dtype
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
                f"dense type is not recognized. Currently {fill_missing=} and {dense.dtype=} and {dense.dtype.type=}")
        return filler, dtype

    def _dense_to_sparse(self, dense: np.ndarray, coords: Dict[str, np.ndarray]) -> 'sp.COO':
        """
        Convert a dense array to a sparse COO array.

        Args:
            dense: The dense array to convert.
            coords: The coordinates of the array.

        Returns:
            A sparse COO array.

        Example:

            ```python
            >>> dense = np.array([[10., 0.], [0., 20.]])
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=dense, coords=coords)
            >>> arr._dense_to_sparse(dense, coords)
            <COO: shape=(2, 2), dtype=int64, nnz=2, fill_value=0>

            ```
        """
        if len(coords) == 0:
            return sp.COO(data=dense, coords=[0], shape=(1,))
        filler, dtype = self._filler_and_dtype(dense, self.fill_value)
        mask = dense != filler
        coords_list = [np.where(mask)[i] for i in range(len(coords))]
        data = dense[mask]
        return sp.COO(coords=coords_list, data=data.astype(dtype), shape=dense.shape, fill_value=filler)

    def _sparse_to_dense(self, sparse: 'sp.COO', coords: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert a sparse COO array to a dense array.

        Args:
            sparse: The sparse COO array to convert.
            coords: The coordinates of the array.

        Returns:
            A dense array.

        Example:
            ```python
            >>> sparse = sp.COO(data=[10, 20], coords=[[0, 1], [0, 1]], shape=(2, 2))
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=sparse, coords=coords)
            >>> arr._sparse_to_dense(sparse, coords)
            array([[10,  0],
                   [ 0, 20]])

            ```
        """
        if len(coords) == 0:
            return sparse.data
        filler, dtype = self._filler_and_dtype(sparse.data, self.fill_value)
        dense = np.full(shape=sparse.shape, fill_value=filler, dtype=dtype)
        dense[tuple(sparse.coords)] = sparse.data
        return dense

    @staticmethod
    def _reorder(self_data: Union[np.ndarray, 'sp.COO'], self_coords: Dict[str, np.ndarray], reorder: List[str] = None) -> Dict[str, Union[np.ndarray, 'sp.COO', Dict[str, np.ndarray]]]:
        """
        Reorder the dimensions of a dense array and its coordinates.

        Args:
            self_data: The dense or sparse array to reorder.
            self_coords: The coordinates of the array.
            reorder: The desired order of dimensions.

        Returns:
            A dictionary containing the reordered dense or sparse array and coordinates.

        Example:
            ```python
            >>> dense = np.array([[10., 0.], [0., 20.]])
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> Array._reorder(dense, coords, reorder=['dim2', 'dim1'])
            {'data': array([[10,  0],
                    [ 0, 20]]), 'coords': {'dim2': [1, 2], 'dim1': ['a', 'b']}}

            ```
        """
        assert reorder is not None, "order must be provided"
        assert set(reorder) == set(self_coords), "order must be equal to self.dims, the order can be different, though"
        if tuple(self_coords) == tuple(reorder):
            return dict(data=self_data, coords=self_coords)
        coords = {k: self_coords[k] for k in reorder}
        data = np.transpose(self_data, axes=[list(self_coords).index(dim) for dim in reorder])
        return dict(data=data, coords=coords)

    def reorder(self, reorder: List[str] = None) -> 'Array':
        """
        Reorder the dimensions of the Array.

        Args:
            reorder: The desired order of dimensions.

        Returns:
            A new Array with the reordered dimensions.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.reorder(reorder=['dim2', 'dim1'])
            Array(data=array([[10.,  0.],
                   [ 0., 20.]]), coords={'dim2': array([1, 2]), 'dim1': array(['a', 'b'], dtype=object)})

            ```
        """
        return Array(**self._reorder(self.data, self.coords, reorder))

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

            ```
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
            >>> coords1 = {'dim1': ['a', 'b']}
            >>> arr1 = Array(data=np.array([10, 20]), coords=coords1)
            >>> coords2 = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[10], [20]]), coords=coords2)
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
            common_dims = [i for i in self.dims if i in common_dims]
            assert len(common_dims) > 0, "At least one dimension must be common"
            uncommon_dims = [i for i in self.dims + other.dims if i not in common_dims]
            uncommon_self = [dim for dim in self.dims if dim in uncommon_dims]
            uncommon_other = [dim for dim in other.dims if dim in uncommon_dims]
            if not self.allow_broadcast:
                assert not all([len(uncommon_self) > 0, len(uncommon_other) > 0]
                           ), f"It is not allowed to have both arrays with uncommon dims. You can apply .expand in one array before performing this operation. {uncommon_self=} {uncommon_other=}"
            if preferred_order is None:
                dims =  list(uncommon_dims) + list(common_dims)
                return dims
            else:
                ordered_uncommon = self._order_with_preference(uncommon_dims, preferred_order)
                ordered_common = self._order_with_preference(common_dims, preferred_order)
                ordered = ordered_uncommon + ordered_common
                return ordered
        else:
            raise Exception(f"This case of ordering is not yet implemented. {self.dims=} {other.dims=}")

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
            >>> coords1 = {'dim1': ['a', 'b']}
            >>> arr1 = Array(data=np.array([10, 20]), coords=coords1)
            >>> coords2 = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[10], [20]]), coords=coords2)
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
                            coords[dim] = np.union1d(self.coords[dim], other.coords[dim])
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
                        coords[dim] = np.union1d(self.coords[dim], other.coords[dim])
                else:
                    self_coords_bool.append(True)
                    coords[dim] = self.coords[dim]
            elif dim in other.coords:
                other_coords_bool.append(True)
                coords[dim] = other.coords[dim]
            else:
                raise Exception(f"Dimension {dim} not found in either arrays")
        return (self_coords_bool, other_coords_bool, coords)

    def _match_coords_change(self, uniondims: List[str], unioncoords: Dict[str, np.ndarray], coords_bool: bool) -> np.ndarray:
        """
        Get the raw dense array based on the union dimensions and coordinates.

        Args:
            uniondims: The list of dimensions in the union.
            unioncoords: The union of coordinates.
            coords_bool: A boolean flag indicating if the coordinates remain the same.

        Returns:
            The raw dense array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> uniondims = ['dim1', 'dim2']
            >>> unioncoords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> coords_bool = True
            >>> arr._match_coords_change(uniondims, unioncoords, coords_bool)
            array([[10,  0],
                   [ 0, 20]])

            ```
        """
        self_dims = [d for d in uniondims if d in self.dims]

        uint_map = {uinfo.max: udtype for uinfo, udtype in map(lambda x: (np.iinfo(x), x), [np.uint8, np.uint16, np.uint32, np.uint64][::-1])}
        if coords_bool:
            if tuple(self.dims) == tuple(self_dims):
                return self.data
            else:
                return np.transpose(self.data, axes=[self.dims.index(dim) for dim in self_dims])
        else:
            if self.data_obj == 'dense':
                old_indices = np.unravel_index(np.arange(self.dense.size, dtype=[uint_map[max_val] for max_val in uint_map if self.dense.size <= max_val][-1]), self.dense.shape)
                old_dense_flattened = self.dense.reshape(-1)
                new_coords = {d: unioncoords[d] for d in self_dims}
                translation_idx = tuple([np.argsort(k)[np.searchsorted(k, self.coords[dim], sorter=np.argsort(k))] for dim, k in new_coords.items()])
                new_shape = self._shape(new_coords)
                filler, dtype = self._filler_and_dtype(self.dense, self.fill_value)
                new_dense = np.full(new_shape, fill_value=filler, dtype=dtype)
                new_indices = tuple([translation_idx[i][old_indices[self.dims.index(dim)]] for i, dim in enumerate(self_dims)])
                new_dense[new_indices] = old_dense_flattened
                return new_dense
            elif self.data_obj == 'sparse':
                old_indices = self.sparse.coords
                old_dense_flattened = self.sparse.data
                new_coords = {d: unioncoords[d] for d in self_dims}
                translation_idx = tuple([np.argsort(k)[np.searchsorted(k, self.coords[dim], sorter=np.argsort(k))] for dim, k in new_coords.items()])
                new_shape = self._shape(new_coords)
                filler, dtype = self._filler_and_dtype(self.dense, self.fill_value)
                new_indices = tuple([translation_idx[i][old_indices[self.dims.index(dim)]] for i, dim in enumerate(self_dims)])
                return sp.COO(coords=new_indices, data=old_dense_flattened.astype(dtype), shape=new_shape, fill_value=filler)
            else:
                raise NotImplementedError

    def _pre_operation(self, other: 'Array') -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform pre-operation steps with another array.

        Args:
            other: The other array to perform the operation with.

        Returns:
            A tuple containing the raw arrays and the union of coordinates.

        Example:
            ```python
            >>> coords1 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[30., 0.], [0., 40.]]), coords=coords2)
            >>> arr1._pre_operation(arr2)
            (array([[10,  0],
                    [ 0, 20]]), array([[30,  0],
                    [ 0, 40]]), {'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        uniondims = self._union_dims(other, preferred_order=self.order)
        self_coords_bool, other_coords_bool, unioncoords = self._union_coords(other, uniondims)
        self_raw_dense = self._match_coords_change(uniondims, unioncoords, all(self_coords_bool))
        other_raw_dense = other._match_coords_change(uniondims, unioncoords, all(other_coords_bool))
        return self_raw_dense, other_raw_dense, unioncoords

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[30., 0.], [0., 40.]]), coords=coords2)
            >>> resulting_array = arr1.dense + arr2.dense
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr1._post_operation(resulting_array, coords)
            Array(data=array([[40.,  0.],
                   [ 0., 60.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if len(coords) == 0:
            return Array(data=resulting_array, coords={})
        return Array(data=resulting_array, coords=coords)

    def _operation(self, self_array: Union[np.ndarray, 'sp.COO'], other_array: Union[np.ndarray, 'sp.COO'], operation: str) -> np.ndarray:
        """
        Perform a math operation on two arrays.

        Args:
            self_array: The first array.
            other_array: The second array.
            operation: The operation to perform.

        Returns:
            The result of the operation.
        """
        return getattr(self_array, operation)(other_array)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[30., 0.], [0., 40.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self._operation(self_arr, other_arr, '__add__')
            return Array(data=arr, coords=coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[30., 0.], [0., 40.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self._operation(self_arr, other_arr, '__mul__')
            return Array(data=arr, coords=coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[30., 0.], [0., 40.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self._operation(self_arr, other_arr, '__sub__')
            return Array(data=arr, coords=coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[30., 40.], [50., 60.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self._operation(self_arr, other_arr, '__truediv__')
            return Array(data=arr, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> 5 + arr
            Array(data=array([[15.,  5.],
                   [ 5., 25.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data + other
            return Array(data=arr, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> 5 * arr
            Array(data=array([[ 50.,   0.],
                   [  0., 100.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = self.data * other
            return Array(data=arr, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> 5 - arr
            Array(data=array([[ -5.,   5.],
                   [  5., -15.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = -self.data + other
            return Array(data=arr, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10., 20.], [30., 40.]]), coords=coords)
            >>> 100 / arr
            Array(data=array([[10.        ,  5.        ],
                   [ 3.33333333,  2.5       ]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other / self.data
            return Array(data=arr, coords=self.coords)

    def __neg__(self) -> 'Array':
        """
        Negate the Array.

        Returns:
            A new Array object with the negated values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> -arr
            Array(data=array([[-10.,   0.],
                   [  0., -20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return Array(data=-self.data, coords=self.coords)

    def __pos__(self) -> 'Array':
        """
        Apply the unary positive operator to the Array.

        Returns:
            A new Array object with the same values.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> +arr
            Array(data=array([[10.,  0.],
                   [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return Array(data=+self.data, coords=self.coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr == other_arr
            return Array(data=arr, coords=coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr != other_arr
            return Array(data=arr, coords=coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr < other_arr
            return Array(data=arr, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> 5 < arr
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other < self.data
            return Array(data=arr, coords=self.coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr <= other_arr
            return Array(data=arr, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> 10 <= arr
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other <= self.data
            return Array(data=arr, coords=self.coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords1)
            >>> coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords2)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr > other_arr
            return Array(data=arr, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 20.], [20., 10.]]), coords=coords)
            >>> 15 > arr
            Array(data=array([[ True, False],
                   [False,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, (int, float)):
            arr = other > self.data
            return Array(data=arr, coords=self.coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr >= other_arr
            return Array(data=arr, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> 10 >= arr
            Array(data=array([[ True,  True],
                   [ True, False]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        arr = other >= self.data
        return Array(data=arr, coords=self.coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr & other_arr
            return Array(data=arr, coords=coords)

    def __rand__(self, other: bool) -> 'Array':
        """
        Perform element-wise logical AND operation between a boolean and the Array (reverse AND).

        Args:
            other: The boolean to perform the operation with.

        Returns:
            A new Array object with the result of the logical AND operation.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> True & arr
            Array(data=array([[0., 0.],
                   [0., 0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, bool):
            arr = other & self.data
            return Array(data=arr, coords=self.coords)

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
            >>> arr1 = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr2 = Array(data=np.array([[20., 0.], [0., 10.]]), coords=coords)
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
            return Array(data=arr, coords=self.coords)
        elif isinstance(other, Array):
            self_arr, other_arr, coords = self._pre_operation(other)
            arr = self_arr | other_arr
            return Array(data=arr, coords=coords)

    def __ror__(self, other: bool) -> 'Array':
        """
        Perform element-wise logical OR operation between a boolean and the Array (reverse OR).

        Args:
            other: The boolean to perform the operation with.

        Returns:
            A new Array object with the result of the logical OR operation.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> True | arr
            Array(data=array([[11.,  1.],
                   [ 1., 21.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        if isinstance(other, bool):
            arr = other | self.data
            return Array(data=arr, coords=self.coords)

    def __invert__(self) -> 'Array':
        """
        Perform element-wise logical NOT operation on the Array.

        Returns:
            A new Array object with the result of the logical NOT operation.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> ~arr
            Array(data=array([[-11.,  -1.],
                   [ -1., -21.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return Array(data=~self.data, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.all()
            False

            ```
        """
        return self.data.all()

    def to_dict(self, rows: Optional[slice] = None, dense: bool = False) -> Dict[str, np.ndarray]:
        """
        Convert the Array to a dictionary.

        Args:
            rows: A slice object specifying the rows to include in the dictionary.
            dense: Whether to convert the dense representation of the Array.

        Returns:
            A dictionary representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.to_dict(dense=True)
            {'dim1': array(['a', 'a', 'b', 'b'], dtype=object), 'dim2': array([1, 2, 1, 2]), 'value': array([10.,  0.,  0., 20.])}

            ```
        """
        if rows is not None:
            if rows.stop > self.capacity:
                rows = slice(rows.start or 0, self.capacity, rows.step or 1)
            else:
                rows = slice(rows.start or 0, rows.stop, rows.step or 1)

        if dense:
            return dict(self.dindex(rows))
        else:
            return dict(self.sindex(rows))

    def to_pandas(self, rows: Optional[slice] = None, dense: bool = None) -> 'pd.DataFrame':
        """
        Convert the Array to a pandas DataFrame.

        Args:
            rows: A slice object specifying the rows to include in the DataFrame.
            dense: Whether to convert the Array to a dense DataFrame. If None, use the default setting.

        Returns:
            A pandas DataFrame representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> df = arr.to_pandas(dense=True)
            >>> df
              dim1  dim2  value
            0    a     1   10.0
            1    a     2    0.0
            2    b     1    0.0
            3    b     2   20.0

            ```
        """

        return pd.DataFrame(self.to_dict(rows=rows, dense=dense or self.dense_dataframe))

    def to_polars(self, rows: Optional[slice] = None, dense: bool = None) -> 'pl.DataFrame':
        """
        Convert the Array to a polars DataFrame.

        Args:
            rows: A slice object specifying the rows to include in the DataFrame.
            dense: Whether to convert the Array to a dense DataFrame. If None, use the default setting.

        Returns:
            A polars DataFrame representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> df = arr.to_polars(dense=False)
            >>> df
            shape: (2, 3)
            ┌──────┬──────┬───────┐
            │ dim1 ┆ dim2 ┆ value │
            │ ---  ┆ ---  ┆ ---   │
            │ str  ┆ i32  ┆ f64   │
            ╞══════╪══════╪═══════╡
            │ a    ┆ 1    ┆ 10.0  │
            │ b    ┆ 2    ┆ 20.0  │
            └──────┴──────┴───────┘

            ```
        """
        return pl.from_dict(self.to_dict(rows=rows, dense=dense or self.dense_dataframe))

    def to_dataframe(self, rows: Optional[slice] = None, dense: bool = None, engine: str = 'pandas') -> Union['pd.DataFrame', 'pl.DataFrame']:
        """
        Convert the Array to a DataFrame using the specified library.

        Args:
            rows: A slice object specifying the rows to include in the DataFrame.
            dense: Whether to convert the Array to a dense DataFrame. If None, use the default setting.
            engine: The library to use for creating the DataFrame. Can be 'pandas' or 'polars'.

        Returns:
            A DataFrame representing the Array, created using the specified library.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> df = arr.to_dataframe(dense=False, engine='pandas')
            >>> df
              dim1  dim2  value
            0    a     1   10.0
            1    b     2   20.0

            ```
        """
        assert engine in ['pandas', 'polars']
        if engine == "pandas":
            return self.to_pandas(rows=rows, dense=dense or self.dense_dataframe)
        elif engine == "polars":
            return self.to_polars(rows=rows, dense=dense or self.dense_dataframe)

    def to_arrow(self) -> 'pa.Table':
        """
        Convert the Array to an Apache Arrow Table.

        Returns:
            An Apache Arrow Table representing the Array.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10.5, 0.], [0., 20.]]), coords=coords)
            >>> table = arr.to_arrow()
            >>> table
            pyarrow.Table
            dim1: string
            dim2: int32
            value: double
            ----
            dim1: [["a","b"]]
            dim2: [[1,2]]
            value: [[10.5,20]]

            ```
        """
        table = pa.Table.from_pydict(self.to_dict(dense=False))
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> new_arr = arr.shrink(dim1=['a'], dim2=[1])
            >>> new_arr
            Array(data=array([[10.]]), coords={'dim1': array(['a'], dtype=object), 'dim2': array([1])})

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
        arr = self.data
        for i, dim in enumerate(self.dims):
            if dim in kwargs:
                if self.data_obj == "dense":
                    arr = np.take(arr, np.argsort(self.coords[dim])[np.searchsorted(self.coords[dim], kwargs[dim], sorter=np.argsort(self.coords[dim]))], axis=self.dims.index(dim))
                elif self.data_obj == "sparse":
                    slc = [slice(None) for _ in self.dims]
                    slc[i] = np.argsort(self.coords[dim])[np.searchsorted(self.coords[dim], kwargs[dim], sorter=np.argsort(self.coords[dim]))]
                    arr = arr[tuple(slc)]
                else:
                    raise ValueError("Unknown data_obj type")
        return Array(data=arr, coords=new_coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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
        data = self._match_coords_change(self.dims, coords, False)
        return Array(data=data, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> reduced_arr = arr.reduce('dim1', aggfunc='sum')
            >>> reduced_arr
            Array(data=array([10., 20.]), coords={'dim2': array([1, 2])})

            ```
        """
        assert dim in self.dims, f"dim {dim} not in self.dims: {self.dims}"
        if isinstance(aggfunc, str):
            assert aggfunc in ['sum', 'mean', 'prod'], "String options for aggfunc can be 'sum', 'mean' or 'prod'"
            if aggfunc == 'sum':
                aggfunc = np.add.reduce
            elif aggfunc == 'mean':
                aggfunc = np.mean
            elif aggfunc == 'prod':
                aggfunc = np.multiply.reduce
        elif isinstance(aggfunc, Callable):
            pass
        data = aggfunc(self.data, axis=self.dims.index(dim))
        dims = [d for d in self.dims if d != dim]
        coords = {k: v for k, v in self.coords.items() if k in dims}
        return Array(data=data, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> shifted_arr = arr._shift_one_dim('dim1', count=1, fill_value=0)
            >>> shifted_arr
            Array(data=array([[ 0.,  0.],
                   [10.,  0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        # TODO: pending sparse implementation
        if fill_value is None:
            fill_value, dtype = self._filler_and_dtype(self.dense, self.fill_value)
            raw_data = self.dense.astype(dtype)
        else:
            raw_data = self.dense
        ax = self.dims.index(dim)
        dense = np.roll(raw_data, shift=count, axis=ax)
        if count > 0:
            dense[tuple(slice(None) if i != ax else slice(0, count) for i in range(dense.ndim))] = fill_value
        elif count < 0:
            dense[tuple(slice(None) if i != ax else slice(count, None) for i in range(dense.ndim))] = fill_value
        return Array(data=dense, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> rolled_arr = arr._roll_one_dim('dim1', count=1)
            >>> rolled_arr
            Array(data=array([[ 0., 20.],
                   [10.,  0.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        # TODO: pending sparse implementation
        assert dim in self.dims, f"{dim} not in dims: {self.dims}"
        assert isinstance(count, int), f"{count} must be int"
        ax = self.dims.index(dim)
        data = np.roll(self.dense, shift=count, axis=ax)
        return Array(data=data, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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

    def insert(self, batch_size: int = None, **kwargs: Union[str, int, float, Dict[str, Union[Dict[str, Any], List[Union[List[str], List[Any]]]]]]) -> 'Array':
        """
        Insert new dimensions into the Array. There are four groups of values that can be inserted:

        - str, int, float

        - dict of dict or dict of list

        - list of str that represents current dimensions

        - np.dtype, type

        Case 1: str, int, float are valid for non-empty arrays. In this case, we insert a new dimension with only one element.
        Case 2 and 3: dict of dict or dict of list are valid for non-empty arrays. In this case, we insert a new dimension which is mapped based on a existing dimension coordinates.
        Case 4: list of str are valid for non-empty arrays. In this case, we insert a new dimension with the concatenation of elements of the corresponding dims in the list.

        Args:
            **kwargs: Keyword arguments specifying the new dimensions and their values.

        Returns:
            A new Array object with the inserted dimensions.

        Example:
            ```python
            >>> # Case 1: str, int, float

            >>> arr = Array(data=np.array([[1., 0.], [0., 2.]]), coords={'dim1': ['a', 'b'], 'dim2': [1, 2]})

            >>> new_arr = arr.insert(dim3='c')

            >>> new_arr
            Array(data=array([[[1., 0.],
                    [0., 2.]]]), coords={'dim3': array(['c'], dtype=object), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            >>> # Case 2: dict of dict

            >>> arr = Array(data=np.array([[1., 0.], [0., 2.]]), coords={'dim1': ['a', 'b'], 'dim2': [1, 2]})

            >>> new_arr = arr.insert(dim3={'dim1': {'a': 'c', 'b': 'c'}})

            >>> # Case 3: dict of list

            >>> new_arr = arr.insert(dim3={'dim1': [['a', 'b'], ['c', 'c']]})

            >>> new_arr
            Array(data=array([[[1., 0.],
                    [0., 2.]]]), coords={'dim3': array(['c'], dtype=object), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            >>> # Case 4: list of dims (str)

            >>> arr = Array(data=np.array([[1., 0.], [0., 2.]]), coords={'dim1': ['a', 'b'], 'dim2': [1, 2]})

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

            ```
        """
        # TODO: add sparse arrays support
        coords = {}
        for new_dim in kwargs:
            assert new_dim not in self.dims, f"new dimension name '{new_dim}' must not exist in the existing dimensions"
            value = kwargs[new_dim]
            # dtype or type only works for for empty arrays. In order to insert a dimension with a dtype, we need to create an empty array

            if isinstance(value, str):
                coords[new_dim] = np.array([value], dtype=np.object_)
            elif isinstance(value, int):
                coords[new_dim] = np.array([value], dtype=np.int32)
            elif isinstance(value, float):
                coords[new_dim] = np.array([value], dtype=np.float32)
            elif isinstance(value, dict):
                assert len(value) == 1, f"Value associated with '{new_dim}' must be a dict with one key. Got {value}"
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
                        kwargs[new_dim][existing_dim][0] = _test_type_and_update(value[existing_dim][0])
                    assert isinstance(
                        kwargs[new_dim][existing_dim][0], np.ndarray), f"Value associated with '{new_dim}' and '{existing_dim}' must be a numpy array. Got {type(kwargs[new_dim][existing_dim][0])}"
                    new_dim_items = value[existing_dim][1]
                    new_dim_items_set = set(new_dim_items)
                    if len(new_dim_items) == len(new_dim_items_set):
                        coords[new_dim] = _test_type_and_update(value[existing_dim][1])
                    else:
                        coords[new_dim] = np.unique(_test_type_and_update(value[existing_dim][1]))
            # this is a list of strings that represent several dimensions named in self.dims. The new dimension is the concatenation of the selected dimensions
            elif isinstance(value, list):
                assert value, "List cannot be empty"
                assert all([isinstance(item, str) for item in value]), "All items in the list must be str"
                assert all([dim in self.dims for dim in value]), "All items in the list must be in dims"
                selected_coords = {dim: self.coords[dim] for dim in value}
                arrays = np.unravel_index(np.arange(self._capacity(selected_coords)), self._shape(selected_coords))
                index = {dim: self.coords[dim][idx] for dim, idx in zip(selected_coords, arrays)}
                coords[new_dim] = _join_str(list(index.values()), sep=":")
                kwargs[new_dim] = {tuple(value): [selected_coords, coords[new_dim]]}
                raise NotImplementedError("This feature is not implemented yet")
            else:
                raise AssertionError(f"Unexpected type: {type(value)}")
        for dim in self.coords:
            coords[dim] = self.coords[dim]

        dense = np.expand_dims(self.dense, axis=tuple(range(len(kwargs))))
        dense = np.broadcast_to(dense, self._shape(coords))

        for new_dim, value in kwargs.items():
            if isinstance(value, dict):
                existing_dim = next(iter(value))
                mapping = {}
                mapping[existing_dim] = np.argsort(coords[existing_dim])[np.searchsorted(coords[existing_dim], value[existing_dim][0], sorter=np.argsort(coords[existing_dim]))]
                mapping[new_dim] = np.argsort(coords[new_dim])[np.searchsorted(coords[new_dim], value[existing_dim][1], sorter=np.argsort(coords[new_dim]))]
                batch_size = batch_size or max(1_000_000, dense.size // 100)
                dense = unravel_index_batch(dense=dense, batch_size=batch_size, coords=coords, existing_dim=existing_dim, new_dim=new_dim, map_relationships=mapping)
        return Array(data=dense, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> new_arr = arr.add_dim(x=0)
            >>> new_arr
            Array(data=array([[[10,  0],
                    [ 0, 20]]]), coords={'x': array([0]), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self.insert(**kwargs)

    def rename(self, **kwargs: Dict[str,Any]) -> 'Array':
        """
        Rename dimensions of the Array.

        Args:
            **kwargs: Keyword arguments specifying the old dimension names and their new names.

        Returns:
            A new Array object with the renamed dimensions.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
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

    def drop(self, dims: Union[str, List[str]], reduce_ok: bool = False) -> 'Array':
        """
        Drop specified dimensions from the Array.

        Args:
            dims: A single dimension or a list of dimensions to drop.
            reduce_ok: Whether to allow reducing the dimensions with non-1 shape. If False, an AssertionError is raised.

        Returns:
            A new Array object with the specified dimensions dropped.

        Example:
            ```python
            >>> coords = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 20.]]), coords=coords)
            >>> new_arr = arr.drop('dim1')
            >>> new_arr
            Array(data=array([10., 20.]), coords={'dim2': array([1, 2])})

            ```
        """
        if isinstance(dims, str):
            dims = [dims]
        assert all(dim in self.dims for dim in dims), f"Dimensions {dims} must be in dims {self.dims}"
        axis = tuple(self.dims.index(dim) for dim in dims)
        if all(self.shape[ax] == 1 for ax in axis):
            data = np.squeeze(self.data, axis=axis)
        else:
            if reduce_ok:
                data = np.add.reduce(self.data, axis=axis, keepdims=False)
            else:
                raise AssertionError("Cannot drop dimensions with non-1 shape. Reduce the dimension either by reduce_ok True or using reduce('dim')")
        coords = {dim: self.coords[dim] for dim in self.dims if dim not in dims}
        return Array(data=data, coords=coords)

    def replacena(self, value) -> 'Array':
        """
        Replace missing values (NaN) in the Array with the specified value.

        Args:
            value: The value to replace NaN with.

        Returns:
            A new Array object with NaN values replaced.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[np.nan, 0], [0, 20]]), coords=coords)
            >>> new_arr = arr.replacena(value=0)
            >>> new_arr
            Array(data=array([[ 0.,  0.],
                   [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        mask = np.isnan(self.dense)
        dense = self.dense.copy()
        dense[mask] = value
        return Array(data=dense, coords=self.coords)

    def replaceinf(self, pos: bool = False, neg: bool = False, value: float = 0) -> 'Array':
        """
        Replace infinite values (inf or -inf) in the Array with the specified value.

        Args:
            pos: Whether to replace positive infinity values.
            neg: Whether to replace negative infinity values.
            value: The value to use for infinite values.

        Returns:
            A new Array object with infinite values replaced.

        Example:
            ```python
            >>> coords = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[np.inf, 20]]), coords=coords)
            >>> new_arr = arr.replaceinf(pos=True)
            >>> new_arr
            Array(data=array([[ 0., 20.]]), coords={'dim1': array(['a'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        assert any([pos, neg]), "pos and neg cannot be both False"
        mask = np.ones_like(self.dense, dtype=bool)
        if pos:
            mask &= np.isinf(self.dense)
        if neg:
            mask &= np.isneginf(self.dense)
        dense = self.dense.copy()
        dense[mask] = value
        return Array(data=dense, coords=self.coords)

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
            >>> arr = Array(data=np.array([[10.123, 0.], [0., 20.456]]), coords=coords)
            >>> rounded_arr = arr.round(decimals=1)
            >>> rounded_arr
            Array(data=array([[10.1,  0. ],
                   [ 0. , 20.5]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return Array(data=self.data.round(decimals=decimals), coords=self.coords)

    def elems_to_datetime(self, new_dim: str, actual_dim: str, reference_date: str, freq: str, sort_coords: bool = True) -> 'Array':
        """
        Convert elements of a dimension to datetime values and create a new dimension.

        Args:
            new_dim: The name of the new dimension to create.
            actual_dim: The name of the existing dimension to convert.
            reference_date: The reference date to start the datetime range from.
            freq: The frequency of the datetime range.
            sort_coords: Whether to sort the coordinates of the new dimension.

        Returns:
            A new Array object with the datetime dimension added.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> new_arr = arr.elems_to_datetime(new_dim='date', actual_dim='dim2', reference_date='2022-01-01', freq='D')
            >>> new_arr
            Array(data=array([[[10.,  0.],
                    [ 0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.],
                    [ 0., 20.]]]), coords={'date': array(['2022-01-01T00:00:00.000000000', '2022-01-02T00:00:00.000000000'],
                  dtype='datetime64[ns]'), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        assert actual_dim in self.dims
        start_date = pd.to_datetime(reference_date)
        t = pd.date_range(start=start_date, periods=self.coords[actual_dim].size, freq=freq)
        if sort_coords:
            return self.insert(**{new_dim: {actual_dim: [np.sort(self.coords[actual_dim]), t]}})
        else:
            return self.insert(**{new_dim: {actual_dim: [self.coords[actual_dim], t]}})

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
            >>> coords = {'dim1': ['a', 'b'], 'dim2': ['1', '2']}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> new_arr = arr.elems_to_int(new_dim='int_dim', actual_dim='dim2')
            >>> new_arr
            Array(data=array([[[10.,  0.],
                    [ 0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.],
                    [ 0., 20.]]]), coords={'int_dim': array([1, 2]), 'dim1': array(['a', 'b'], dtype=object), 'dim2': array(['1', '2'], dtype=object)})

            ```
        """
        serie = pd.Series(data=self.coords[actual_dim])
        serie = serie.str.extract(r"(\d+)", expand=False).astype("int")
        new_array = self.insert(**{new_dim: {actual_dim: [self.coords[actual_dim], serie.values]}})
        return new_array

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
            >>> arr = Array(data=np.array([[0.3, 0.4], [0.7, 0.6]]), coords=coords)
            >>> chosen_arr = arr.choice(dim='dim1', seed=42)
            >>> chosen_arr
            Array(data=array([[False, False],
                   [ True,  True]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        rng = np.random.default_rng(seed=seed)
        assert dim in self.dims, f"dim {dim} not in self.dims: {self.dims}"
        axis = self.dims.index(dim)
        probabilities = self.dense
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> expanded_arr = arr.expand(new_dim=['x','y'])
            >>> expanded_arr.reorder(reorder=['dim1', 'dim2', 'new_dim'])
            Array(data=array([[[10., 10.],
                    [ 0.,  0.]],
            <BLANKLINE>
                   [[ 0.,  0.],
                    [20., 20.]]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2]), 'new_dim': array(['x', 'y'], dtype=object)})

            ```
        """
        coords = {}
        for dim in kwargs:
            assert dim not in self.dims, f"dim {dim} already exists in self.dims: {self.dims}"
            coords[dim] = _test_type_and_update(kwargs[dim])
        for dim in self.coords:
            if dim not in coords:
                coords[dim] = self.coords[dim]
        expanded = np.expand_dims(self.dense, axis=tuple(range(len(kwargs))))
        dense = np.broadcast_to(expanded, self._shape(coords))
        return Array(data=dense, coords=coords)

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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> result_arr = arr.ufunc(dim='dim1', func=np.sum, keepdims=True)
            >>> result_arr
            Array(data=array([[10., 20.],
                   [10., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        axis = self.dims.index(dim)
        result = func(self.dense, axis=axis, keepdims=keepdims, **func_kwargs)
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
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> arr.df
              dim1  dim2  value
            0    a     1   10.0
            1    b     2   20.0

            ```
        """
        if self.engine == "pandas":
            return self.to_pandas(dense=self.dense_dataframe)
        elif self.engine == "polars":
            return self.to_polars(dense=self.dense_dataframe)
        else:
            raise Exception("ka.settings.engine must be either 'pandas' or 'polars'")

    def todense(self) -> np.ndarray:
        """
        Returns the dense representation of the Array.

        Returns:
            A dense numpy array.
        """
        return self.dense
        
    def tensordot(self, other: 'Array', allow_broadcast: bool=True) -> 'Array':
        """
        Perform a tensor dot product between two Array objects.

        Args:
            other: The second Array object to perform the tensor dot product with.
            allow_broadcast: Whether to allow broadcasting during the tensor dot product.

        Returns:
            A new Array object with the tensor dot product applied.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> other = Array(data=np.array([[10., 0.], [0., 20.]]), coords=coords)
            >>> result_arr = arr.tensordot(other)
            >>> result_arr
            Array(data=array(500.), coords={})

            ```
        """
        common_dims = list(set(self.dims).intersection(set(other.dims)))
        common_len = len(common_dims)
        assert common_len > 0, "At least one dimension must be common"
        memory = self.allow_broadcast
        self.allow_broadcast = allow_broadcast
        self_arr, other_arr, coords = self._pre_operation(other)
        self_dims_idx = [i for i in range(self_arr.ndim)]
        other_dims_idx = [i for i in range(other_arr.ndim)]
        self.allow_broadcast = memory
        axes = tuple([self_dims_idx[-common_len:], other_dims_idx[-common_len:]])
        arr = np.tensordot(self_arr, other_arr, axes=axes)
        new_coords = {dim: coords[dim] for dim in coords if dim not in common_dims}
        return Array(data=arr, coords=new_coords)
    
    def tensorsolve(self, other: 'Array') -> 'Array':
        """
        Perform a tensor solve between two Array objects.

        Args:
            other: The second Array object to perform the tensor solve with.

        Returns:
            A new Array object with the tensor solve applied.

        Example:
            ```python
            >>> arr = Array(data=np.array([[10., 0.], [0., 20.]]), coords={'dim1': ['a', 'b'], 'dim2': [1, 2]})
            >>> other = Array(data=np.array([10., 0.]), coords={'dim2': [1, 2]})
            >>> result_arr = arr.tensorsolve(other)
            >>> result_arr
            Array(data=array([1., 0.]), coords={'dim1': array(['a', 'b'], dtype=object)})

            ```
        """
        common_dims = list(set(self.dims).intersection(set(other.dims)))
        common_len = len(common_dims)
        assert common_len > 0, "At least one dimension must be common"
        self_arr, other_arr, coords = self._pre_operation(other)
        if self.data_obj == 'sparse':
            self_arr = self_arr.todense()
        if other.data_obj == 'sparse':
            other_arr = other_arr.todense()
        arr = np.linalg.tensorsolve(self_arr, other_arr)
        new_coords = {dim: coords[dim] for dim in coords if dim not in common_dims}
        return Array(data=arr, coords=new_coords)
    
    def dropna(self, value: float=0.0) -> 'Array':
        """
        Replace missing values (NaN) in the Array with the specified value.

        Args:
            value: The value to replace NaN with.

        Returns:
            A new Array object with NaN values replaced.

        Example:
            ```python
            >>> coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[np.nan, 0], [0, 20]]), coords=coords)
            >>> new_arr = arr.dropna(value=0)
            >>> new_arr
            Array(data=array([[ 0.,  0.],
                   [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self.replacena(value=value)
    
    def dropinf(self, pos:bool, neg:bool) -> 'Array':
        """
        Replace infinite values (inf or -inf) in the Array with the specified value.

        Args:
            pos: Whether to replace positive infinity values.
            neg: Whether to replace negative infinity values.

        Returns:
            A new Array object with infinite values replaced.

        Example:
            ```python
            >>> coords = {'dim1': ['a'], 'dim2': [1, 2]}
            >>> arr = Array(data=np.array([[np.inf, 20]]), coords=coords)
            >>> new_arr = arr.dropinf(pos=True, neg=False)
            >>> new_arr
            Array(data=array([[ 0., 20.]]), coords={'dim1': array(['a'], dtype=object), 'dim2': array([1, 2])})

            ```
        """
        return self.replaceinf(pos=pos, neg=neg)



def concat(arrays: List[Array], dim: Union[str,None] = None) -> Array:
    """
    Concatenate multiple Array objects along a new dimension.

    Args:
        arrays: A list of Array objects to concatenate.
        dim: The dimension to concatenate along. If None, the first dimension is used.

    Returns:
        A new Array object with the concatenated arrays.

    Example:
        ```python
        >>> arr1 = Array(data=np.array([[0., 10.]]), coords={'dim1': ['a'], 'dim2': [1, 2]})
        >>> arr2 = Array(data=np.array([[30., 40.]]), coords={'dim1': ['b'], 'dim2': [1, 2]})
        >>> concatenated_arr = concat([arr1, arr2])
        >>> concatenated_arr
        Array(data=array([[ 0., 10.],
               [30., 40.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

        ```
    """
    if len(arrays) == 1:
        return arrays[0]
    dims = arrays[0].dims[:]
    assert all([isinstance(arr, Array) for arr in arrays]), "All list items must be karray.Array"
    assert all([set(arr.dims) == set(dims) for arr in arrays]), "All array must have the same dimensions"

    if dim is not None:
        assert dim in dims, f"dim {dim} not in self.dims: {dims}"
    else:
        dim = dims[0]

    total_axis_elements = sum([arr.shape[dims.index(dim)] for arr in arrays])
    elements = [arr.coords[dim] for arr in arrays]
    for j in range(len(elements)):
        for i in range(j+1, len(elements)):
            assert len(np.intersect1d(elements[j],elements[i],assume_unique=True, return_indices=False)) == 0, "Arrays must have unique elements along the specified axis"
                
    common_coords =  union_multi_coords(*[arr.coords for arr in arrays])
    assert len(common_coords[dim]) == total_axis_elements, f"Total number of elements along {dim} must be equal"
    shape = [common_coords[dim].size for dim in dims]
    init = Array(data=np.zeros(shape), coords=common_coords)
    return functools_reduce(lambda x, y: Array.__add__(x, y), arrays, init)


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
    df = None
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
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10., 20.])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

        ```
    """
    assert with_ in ["pandas", "polars"]
    restored_table = feather.read_table(path, use_threads=use_threads, memory_map=True)
    column_names = restored_table.column_names
    assert "value" in column_names, "Column named 'value' must exist"
    custom_meta_key = 'karray'
    if custom_meta_key.encode() in restored_table.schema.metadata:
        restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
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
        Array(data=array([[10.,  0.],
               [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

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
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10., 20.])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

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
                value = np.loadtxt(path, skiprows=1, delimiter=delimiter, usecols=i, dtype=dtype)
            else:
                index[col] = np.loadtxt(path, skiprows=1, delimiter=delimiter, usecols=i, dtype=dtype)
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
        {'data': ({'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])}, array([10., 20.])), 'coords': {'dim1': ['a', 'b'], 'dim2': [1, 2]}}

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
        Array(data=array([[10.,  0.],
               [ 0., 20.]]), coords={'dim1': array(['a', 'b'], dtype=object), 'dim2': array([1, 2])})

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


def ndarray_choice(p: np.ndarray, axis: int, rng: np.random.Generator) -> np.ndarray:
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

        cum = np.cumsum(p, axis=axis)

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
        nd_dense = flatten_dense.reshape(shape)
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
                        new_coords[dim] = np.union1d(new_coords[dim], coords[dim])
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
            elif issubclass(item.dtype.type, (np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64)):
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
            elif issubclass(value_type, (np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64)):
                selected_type = item[0].dtype
            elif issubclass(value_type, (np.float16, np.float32, np.float64)):
                selected_type = item[0].dtype
            else:
                raise Exception(
                    f"Type: {type(item[0])} not implemented yet. Item: {item}")
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


def _test_type_and_update_value(value: Union[np.ndarray, list, float, int, bool, np.bool_, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]) -> np.ndarray:
    assert isinstance(value, (np.ndarray, list, float, int, bool, np.bool_, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16,
                      np.float32, np.float64)), f"Value attribute must be a numpy array, list, float, int, or bool. Got {type(value)}"
    if isinstance(value, np.ndarray):
        assert issubclass(value.dtype.type, (np.bool_, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
                          ), "dtype suppoerted for value attribute: np.bool_, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64"
        if value.ndim == 0:
            value = value.reshape((value.size,))
    elif isinstance(value, (float, int, bool, np.bool_, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
        value = np.array([value])
    elif isinstance(value, list):
        if value:
            if isinstance(value[0], str):
                raise NotImplementedError(
                    "Data, either dense or sparse object does not support string values for value attribute")
            elif isinstance(value[0], (int, float, bool)):
                value = np.array(value)
            elif isinstance(value[0], np.ndarray):
                if issubclass(value[0].dtype.type, (np.bool_, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)):
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


def unravel_index_batch(dense, batch_size, coords, existing_dim, new_dim, map_relationships: Dict[str, np.ndarray]):
    dense_bins = []
    for i in range(0, dense.size, batch_size):
        batch_indices = np.arange(i,i+batch_size if i+batch_size < dense.size else dense.size)
        indices = np.unravel_index(batch_indices, shape=dense.shape)
        match_idx = map_relationships[new_dim][map_relationships[existing_dim]][indices[list(coords).index(existing_dim)]]
        mask = indices[list(coords).index(new_dim)] != match_idx
        dense_bins_flat = dense[indices]
        dense_bins_flat[mask] = 0.0
        dense_bins.append(dense_bins_flat)
    dense_bins = np.hstack(dense_bins)
    return dense_bins.reshape(dense.shape)


css_style = '''<style>

.details {
    user-select: none;
}

.details > summary {
    display: flex;
    cursor: pointer;
    position: relative;
}

.details > summary .span.icon {
    width: 24px;
    height: 24px;
    transition: all 0.3s;
    margin-left: auto;
}

.details[open] > summary.summary ::-webkit-details-marker {
    display: none;
}

.details[open] > summary .span.icon {
    transform: rotate(180deg);
}

/* Tooltip styles */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted black;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 165px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 4px;
    padding: 2px 0;
    position: absolute;
    z-index: 1;
    font-size: 11px;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
}

.tooltip .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -8px;
    border-width: 8px;
    border-style: solid;
    border-color: #fff transparent transparent transparent;
}

.tooltip-top {
    bottom: 90%;
    margin-left: -40px;
}
</style>'''
