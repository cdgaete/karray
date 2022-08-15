
import numpy as np
import pandas as pd
import itertools
import copy

from .utils import _format_bytes


class array:
    """
    A class for labelled multidimensional arrays.
    """
    def __init__(self, coo:np.array=None, dims:list=None, coords:dict=None, **kwargs):
        self.coo = coo
        self.dims = dims
        self.coords = coords
        self._check_dims_array_order()
        self.preferred_dtype = None
        self.preferred_order = None
        return None

    def __repr__(self):
        return f"Karray.array(coo, coords, dims)"

    def _repr_html_(self):
        if all([arg is not None for arg in [self.coo, self.dims, self.coords]]):
            return ('<h3>karray</h3>'
                '<table>'
                f'<tr><th>coo</th><td>size: {_format_bytes(self.coo.nbytes)}</td></tr>'
                f'<tr><th>dims</th><td>{self.dims}</td></tr>'
                f'<tr><th>coords</th><td>shape {self.shape}</td></tr>'
                '</table>')
        else:
            return ('<h3>karray</h3>'
                '<table>'
                '<tr><th>coo</th><td>None</td></tr>'
                '<tr><th>dims</th><td>None</td></tr>'
                '<tr><th>coords</th><td>None</td></tr>'
                '</table>')

    def __str__(self):
        pass

    @property
    def shape(self):
        return [len(self.coords[dim]) for dim in self.coords]

    @property
    def _cindex(self):
        '''
        Return index of coo array as an array.
        '''
        return self.coo[:,range(len(self.dims))]

    @property
    def cindex(self):
        '''
        Returns the index of coo array as a list of tuples.
        '''
        return list(map(tuple, self._cindex))

    @property
    def dindex(self):
        '''
        Returns the index of dense array as a list of tuples.
        '''
        return self._dindex(self.coords)

    @staticmethod
    def _dindex(coords):
        '''
        Returns the index of dense array as a list of tuples.
        '''
        ordered_coords = {k:list(np.sort(v)) for k,v in coords.items()}
        cartesian_tuples = list(itertools.product(*list(ordered_coords.values())))
        return cartesian_tuples

    def _check_dims_array_order(self):
        if all([arg is not None for arg in [self.coo, self.dims, self.coords]]):
            assert len(self.dims) == len(self.coords), "dims and coords must be equal"
            assert all([all(np.sort(np.unique(arr)) == np.sort(arr)) for arr in self.coords.values()]), "Coordinates must be unique."
            for i in range(len(self.dims)):
                assert set(self._cindex.T[i]).issubset(self.coords[self.dims[i]]), "All elements of index must be included in coords."
        elif any([arg is not None for arg in [self.coo, self.dims, self.coords]]):
            raise ValueError("coo, dims and coords must be all provided.")
        else:
            pass

    def to_dataframe(self, dtype='float64'):
        dense = self.dense(dtype=dtype)
        mi = pd.MultiIndex.from_tuples(self.dindex, names=self.dims)
        df = pd.DataFrame(pd.Series(dense.reshape(-1), index=mi, name='value'))
        return df

    def copy(self):
        return copy.deepcopy(self)

    def dense(self, dtype='float64'):
        '''
        Return a dense version of the array based on the coords.
        
        Returns:
            a index and np.narray.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4],'y':[2,5]})
            >>> dense = ar.dense()
        '''
        values = self.coo[:,len(self.dims)]
        cartesian_tuples = self.dindex
        hshTbl = {}
        ix_zeros = []
        for idx, ai in enumerate(cartesian_tuples):
            hshTbl[ai] = idx
        for bi in self.cindex:
            if bi in hshTbl:
                ix_zeros.append(hshTbl[bi])
        zeros = np.zeros((len(cartesian_tuples),), dtype=dtype)
        zeros[ix_zeros] = values
        return zeros.reshape(self.shape)

    def reorder(self, dims=None):
        '''
        Reorder the array based on the dims.
        
        Args:
            dims (list): list of dimension names.

        Returns:
            coo array, dims and coords are reordered.
        '''
        if dims is None:
            return array(coo=self.coo, dims=self.dims, coords=self.coords)
        else:
            assert set(dims) == set(self.dims), "dims must be equal to self.dims"
            coords = {k:self.coords[k] for k in dims}
            dorder = [self.dims.index(d) for d in dims]
            dim_order = dorder + [len(dorder)]
            coo = self.coo.T[dim_order].T
            return array(coo=coo, dims=dims, coords=coords)

    @staticmethod
    def _order_with_preference(dims:list, preferred_order:list=None):
        if preferred_order is None:
            return dims
        else:
            ordered = []
            unourdered = dims[:]
            for dim in preferred_order:
                if dim in unourdered:
                    ordered.append(dim)
                    unourdered.remove(dim)
            ordered.extend(unourdered)
            return ordered

    def _union_dims(self, other, preferred_order: list = None):
        '''
        Returns the union of the dims of two arrays.

        Args:
            other (karray): other array.
            preferred_order (list): list of dimension names in the preferred order.

        Returns:
            a list of dimension names.
        '''
        if set(self.dims) == set(other.dims):
            return self._order_with_preference(self.dims, preferred_order)
        elif len(set(self.dims).symmetric_difference(set(other.dims))) > 0:
            common_dims = set(self.dims).intersection(set(other.dims))
            assert len(common_dims) > 0, "At least one dimension must be common"
            uncommon_dims = set(self.dims).symmetric_difference(set(other.dims))
            assert not all([uncommon_dims.issubset(self.dims), uncommon_dims.issubset(other.dims)]), "Uncommon dims must be in only one array."
            if len(uncommon_dims) > 1:
                print(f"{common_dims} are dims in common, but more than one are uncommon: {uncommon_dims}.")
            unordered = list(set(self.dims).union(set(other.dims)))
            semi_ordered = self._order_with_preference(unordered, preferred_order)
            ordered_common = []
            if preferred_order is None:
                return list(common_dims) + list(uncommon_dims)
            else:
                for dim in preferred_order:
                    if dim in common_dims:
                        ordered_common.append(dim)
                        common_dims.remove(dim)
                ordered_common.extend(common_dims)
                for dim in ordered_common:
                    if dim in semi_ordered:
                        semi_ordered.remove(dim)
                ordered =  ordered_common + semi_ordered
                return ordered

    def _union_coords(self, other, dims):
        '''
        Returns the union of the coords of two arrays.
        '''

        coords = {}
        for dim in dims:
            if dim in self.coords:
                if dim in other.coords:
                    coords[dim] = sorted(list(set(self.coords[dim]).union(set(other.coords[dim]))))
                else:
                    coords[dim] = self.coords[dim]
            elif dim in other.coords:
                coords[dim] = other.coords[dim]
            else:
                raise Exception(f"Dimension {dim} not found in either array")
        return coords

    def _get_dense(self, other, dims, coords, dtype='float64'):
        inv_dims = dims[::-1]
        self_dims = [d for d in inv_dims if d in self.dims]
        self_coords = {d:coords[d] for d in self_dims}
        self_coo_ordered_array = self.reorder(self_dims)
        self_array = array(coo=self_coo_ordered_array.coo, dims=self_dims, coords=self_coords)
        other_dims = [d for d in inv_dims if d in other.dims]
        other_coords = {d:coords[d] for d in other_dims}
        other_coo_ordered_array = other.reorder(other_dims)
        other_array = array(coo=other_coo_ordered_array.coo, dims=other_dims, coords=other_coords)
        return self_array.dense(dtype), other_array.dense(dtype)

    def _pre_operation_with_array(self, other, preferred_order: list = None, dtype=None):
        if preferred_order is None:
            assert self.preferred_order == other.preferred_order, "'preferred_order' attribute must be equal in both arrays"
            preferred_order_ = self.preferred_order
        else:
            preferred_order_ = preferred_order
        if dtype is None:
            assert self.preferred_dtype == other.preferred_dtype, "'preferred_dtype' attribute must be equal in both arrays"
            dtype_ = self.preferred_dtype or 'float64'
        else:
            dtype_ = dtype

        dims = self._union_dims(other, preferred_order_)
        coords = self._union_coords(other, dims)
        self_dense, other_dense = self._get_dense(other, dims, coords, dtype_)
        return self_dense, other_dense, dims, coords

    def _pre_operation_with_number(self, dtype=None):
        if dtype is None:
            dtype_ = self.preferred_dtype or 'float64'
        else:
            dtype_ = dtype

        self_dense = self.dense(dtype_)
        return self_dense

    def _post_operation_with_array(self, resulting_dense, dims, coords, preferred_order: list = None, dtype=None):
        if preferred_order is None:
            preferred_order_ = self.preferred_order
        else:
            preferred_order_ = preferred_order

        if dtype is None:
            dtype_ = self.preferred_dtype or 'float64'
        else:
            dtype_ = dtype

        dim_reverse = dims[::-1]
        coords_reverse = {dim:coords[dim] for dim in dim_reverse}
        coo_dense = np.hstack([np.array(self._dindex(coords_reverse), dtype=object), resulting_dense.reshape(resulting_dense.size,1)])
        coo = coo_dense[coo_dense[:,-1] != 0]
        inv_array = array(coo = coo, dims = dim_reverse,  coords = coords_reverse)
        inv_array.preferred_order = preferred_order_
        inv_array.preferred_dtype = dtype_
        dims = self._order_with_preference(dims=dims, preferred_order=preferred_order_)
        arr = inv_array.reorder(dims)
        return arr

    def _post_operation_with_number(self, resulting_dense, preferred_order: list = None, dtype=None):
        if preferred_order is None:
            preferred_order_ = self.preferred_order
        else:
            preferred_order_ = preferred_order

        if dtype is None:
            dtype_ = self.preferred_dtype or 'float64'
        else:
            dtype_ = dtype

        coo_dense = np.hstack([np.array(self._dindex(self.coords), dtype=object), resulting_dense.reshape(resulting_dense.size,1)])
        coo = coo_dense[coo_dense[:,-1] != 0]
        arr = array(coo = coo, dims = self.dims,  coords = self.coords)
        arr.preferred_order = preferred_order_
        arr.preferred_dtype = dtype_
        return arr

    def __add__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense + other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense + other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense * other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense * other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense - other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense - other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense / other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense / other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = other + self_dense
            return self._post_operation_with_number(resulting_dense)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = other * self_dense
            return self._post_operation_with_number(resulting_dense)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = other - self_dense
            return self._post_operation_with_number(resulting_dense)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = other / self_dense
            return self._post_operation_with_number(resulting_dense)

    def __neg__(self):
        self_dense = self._pre_operation_with_number()
        resulting_dense = -self_dense
        return self._post_operation_with_number(resulting_dense)

    def __pos__(self):
        self_dense = self._pre_operation_with_number()
        resulting_dense = +self_dense
        return self._post_operation_with_number(resulting_dense)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense ** other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense ** other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = other ** self_dense
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = other_dense ** self_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense < other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense < other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __le__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense <= other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense <= other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense > other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense > other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense >= other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense >= other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense == other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense == other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            self_dense = self._pre_operation_with_number()
            resulting_dense = self_dense != other
            return self._post_operation_with_number(resulting_dense)
        elif isinstance(other, array):
            self_dense, other_dense, dims, coords = self._pre_operation_with_array(other)
            resulting_dense = self_dense != other_dense
            return self._post_operation_with_array(resulting_dense, dims, coords)

    def shrink(self, **kwargs):
        '''
        Shrink the array based on the kwargs.

        Args:
            kwargs (dict): dimension names as keys and a list of coordinates to keep as values.

        Returns:
            a new coo array with the coordinates specified in the kwargs.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4],'y':[2,5]})
            >>> ar = ar.shrink(x=[4])

        '''
        assert set(self.dims) == set(self.coords), "coords keys and dims must be equal"
        assert all([kw in self.coords for kw in kwargs.keys()]), "Selected dimension must be in coords"
        assert all([isinstance(val, list) for val in kwargs.values()]), "Keeping elements must be contained in lists"
        assert all([set(kwargs[kw]).issubset(self.coords[kw]) for kw in kwargs]), "All keeping elements must be included of coords"
        # removing elements from coords dictionary
        new_coords = {k:v[:] for k,v in self.coords.items()}
        for kdim, klist in kwargs.items():
            for cdim, clist in self.coords.items():
                if kdim == cdim:
                    for citem in clist:
                        if citem not in klist:
                            new_coords[cdim].remove(citem)
        # new coo array
        i = 0
        indexes = []
        for ar in self._cindex.T:
            if self.dims[i] in kwargs:
                keep = np.array(kwargs[self.dims[i]], dtype=object)
                indexes.append(np.in1d(ar, keep))
            i+=1
        masksum = sum(indexes)
        mask = masksum == max(masksum)
        coo = self.coo[mask]
        return array(coo=coo, dims=self.dims, coords=new_coords)

    def coords_replace(self, dim:str=None, unique_coords:list=None):
        '''
        Replace the coordinates of a dimension.

        Args:
            dim (str): the dimension to replace.
            unique_coords (list): the new coordinates.

        Returns:
            a new coo array with the coordinates specified in the kwargs.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4],'y':[2,5]})
            >>> ar = ar.coords_replace(dim='x', unique_coords=['a','b'])
        '''
        assert dim in self.dims, "Dimension must be in dims"
        assert isinstance(unique_coords, list), "Unique_coords must be a list"
        assert len(set(unique_coords)) == len(unique_coords), "Unique_coords must be unique"
        assert len(unique_coords) == len(self.coords[dim]), "Unique_coords must be the same length as the original coordinates"

        coo = self.coo.copy()
        coo[:,self.dims.index(dim)] = np.array(unique_coords)[np.array(self.coords[dim]).searchsorted(coo[:,self.dims.index(dim)], sorter=list(range(len(self.coords[dim]))))]
        coords = {k:v for k,v in self.coords.items()}
        coords[dim] = unique_coords
        return array(coo=coo, dims=self.dims, coords=coords)

    def str2int(self, dim:str=None, use_index:bool=False):
        '''
        Convert the coordinates of a dimension to integers if the coordinates are numbered strings.

        Args:
            dim (str): the dimension to convert.

        Returns:
            a new coo array with the coordinates specified in the kwargs.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':['a01','a02'],'y':['c','d']})
            >>> ar = ar.str_to_int(dim='x')
        '''
        assert dim in self.dims, "Dimension must be in dims"
        if use_index:
            int_coords = list(range(len(self.coords[dim])))
        else:
            int_coords = [int(''.join(filter(str.isdigit, elem))) for elem in self.coords[dim]]
            assert len(set(int_coords)) == len(int_coords), "There are duplicate values. You can use use_index=True to use the index instead of getting the numbers from string."
        return self.coords_replace(dim=dim, unique_coords=int_coords)







def from_pandas(df, dims: list=None):
    assert 'value' in df.columns, "Dataframe must have a 'value' column"
    if dims is None:
        if len(df.columns) == 1:
            dims = list(df.index.names)
            assert len(dims) == len(set(dims)), "MultiIndex must have named dimensions."
        else:
            dims = list(df.columns)
            dims.remove('value')
    else:
        if len(df.columns) == 1:
            assert set(dims) == set(list(df.index.names)), "'dims' list must coincide with MultiIndex names."
        else:
            supposed_dims = list(df.columns)
            supposed_dims.remove('value')
            assert set(dims) == set(supposed_dims), "'dims' list must coincide with Dataframe columns."
    
    coords = {}
    for dim in dims:
        if len(df.columns) == 1:
            coords[dim] = list(np.sort(df.index.get_level_values(dim).unique()))
        else:
            coords[dim] = list(np.sort(df[dim].unique()))

    if len(df.columns) == 1:
        dt = df[df['value'] != 0].reset_index()
        col = dt.pop("value")
        dt = dt[dims]
        dt.insert(len(dt.columns), col.name, col)
        coo = dt.to_numpy()
    else:
        dt = df[df['value'] != 0]
        col = dt.pop("value")
        dt = dt[dims]
        dt.insert(len(dt.columns), col.name, col)
        coo = dt.to_numpy()
    return array(coo, dims, coords)