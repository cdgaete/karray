
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import json
from functools import reduce
from itertools import product
from .utils import _format_bytes


class array:
    """
    A class for labelled multidimensional arrays.
    """
    def __init__(self, coo:np.array=None, dims:list=None, coords:dict=None, dtype:str='float64', order:list=None, dense:np.array=None, cindex:list=None, dindex:list=None, **kwargs):
        '''
        Initialize a karray.
        
        Args:
            coo (np.array): a coo array.
            dims (list): list of dimension names.
            coords (dict): dictionary of coordinates.
            kwargs: keyword arguments for the array. Options are: 
                    dtype (str) for resulting dense array, 
                    order (list of strings) for prefered order of dims if they are in the coo array.
        '''
        self.__dict__["_repo"] = {}
        self.coo = coo
        self.dims = dims
        self.coords = None
        self.dtype = dtype
        self.order = order
        self._check_input(coords)
        self.dense_ = dense
        self.cindex_ = cindex
        self.dindex_ = dindex
        oredered_dims = self._order_with_preference(self.dims, self.order)
        ordered_dict = self._reorder(self.coo, self.dims, self.coords, oredered_dims)
        self.coo = ordered_dict['coo']
        self.dims = ordered_dict['dims']
        self.coords = ordered_dict['coords']
        return None

    def _repr_html_(self):
        if all([arg is not None for arg in [self._repo['coo'], self._repo['dims'], self._repo['coords']]]):
            coo_nbytes = _format_bytes(self._repo['coo'].nbytes)
            dims = self._repo['dims']
            shape = self.shape
            return ('<h3>karray</h3>'
                '<table>'
                f'<tr><th>coo</th><td>size: {coo_nbytes}</td></tr>'
                f'<tr><th>dims</th><td>{dims}</td></tr>'
                f'<tr><th>coords</th><td>shape {shape}</td></tr>'
                '</table>')
        else:
            return ('<h3>karray</h3>'
                '<table>'
                '<tr><th>coo</th><td>None</td></tr>'
                '<tr><th>dims</th><td>None</td></tr>'
                '<tr><th>coords</th><td>None</td></tr>'
                '</table>')

    def __setattr__(self, name, value):
        self._repo[name] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name == "cindex_":
            if name in self._repo:
                if self._repo[name] is None:
                    self._repo[name] = self.cindex()
                    return self._repo[name]
                else:
                    return self._repo[name]
        elif name == "dindex_":
            if name in self._repo:
                if self._repo[name] is None:
                    self._repo[name] = self.dindex()
                    return self._repo[name]
                else:
                    return self._repo[name]
        elif name == "dense_":
            if name in self._repo:
                if self._repo[name] is None:
                    self._repo[name] = self.dense()
                    return self._repo[name]
                else:
                    return self._repo[name]
        else:
            return self._repo[name]

    def __repr__(self):
        return f"Karray.array(coo, dims, coords)"

    def _add_coords(self):
        self.coords = {}
        for i, dim in enumerate(self.dims):
            self.coords[dim] = list(np.sort(np.unique(self.coo[:,i])))

    def _check_input(self, coords):
        assert all([arg is not None for arg in [self.coo, self.dims, self.dtype]]), "coo, dims and dtype must be defined"
        assert isinstance(self.coo, np.ndarray), "coo must be a numpy array"
        assert isinstance(self.dims, list), "dims must be a list"
        if coords is None:
            print("coords is None. self.coords has been generated from coo and dims.")
            self._add_coords()
        else:
            assert isinstance(coords, dict), "coords must be a dictionary"
            self.coords = coords
            assert len(self.dims) == len(self.coords), "Length of dims and coords must be equal"
            assert all([all(np.sort(np.unique(arr)) == np.sort(arr)) for arr in self.coords.values()]), "Coordinates must be unique."
            for i in range(len(self.dims)):
                assert set(self._cindex.T[i]).issubset(self.coords[self.dims[i]]), f"All elements of coo array must be included in coords for dim: {self.dims[i]}."

    @property
    def shape(self):
        return [len(self.coords[dim]) for dim in self.coords]

    @property
    def _cindex(self):
        '''
        Return index of coo array as an array.
        '''
        return self.coo[:,range(len(self.dims))]

    def cindex(self):
        '''
        Returns the index of coo array as a list of tuples.
        '''
        return list(zip(*self._cindex.T))

    def dindex(self):
        '''
        Returns the index of dense array as a list of tuples.
        '''
        return self._dindex(self.coords)

    # TODO: This function needs to be optimized.
    def _dindex(self, coords):
        '''
        Returns the index of dense array as a list of tuples.
        '''
        return sorted(product(*list(coords.values())))

    # TODO: This function needs to be optimized.
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
        hshTbl = {}
        ix_zeros = []
        for idx, ai in enumerate(self.dindex_):
            hshTbl[ai] = idx
        for bi in self.cindex_:
            if bi in hshTbl:
                ix_zeros.append(hshTbl[bi])
        zeros = np.zeros((len(self.dindex_),), dtype=dtype)
        zeros[ix_zeros] = values
        return zeros.reshape(self.shape)

    @staticmethod
    def _reorder(self_coo, self_dim, self_coords, order=None):
        '''
        Reorder the array based on the dims.
        
        Args:
            order (list): list of dimension names.

        Returns:
            coo array, dims and coords are reordered.
        '''
        assert order is not None, "order must be provided"
        assert set(order) == set(self_dim), "order must be equal to self.dims, the order can be different, though"
        if self_dim == order:
            return dict(coo=self_coo, dims=self_dim, coords=self_coords)
        coords = {k:self_coords[k] for k in order}
        dorder = [self_dim.index(d) for d in order]
        dim_order = dorder + [len(dorder)]
        coo = self_coo.T[dim_order].T
        return dict(coo=coo, dims=order, coords=coords)

    def reorder(self, order=None):
        return array(dtype=self.dtype, order=None,**self._reorder(self.coo, self.dims, self.coords, order))

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

    @staticmethod
    def _get_inv_dense(obj, commondims, commoncoords):
        self_dims = [d for d in commondims if d in obj.dims]
        self_coords = {d:commoncoords[d] for d in self_dims}
        if obj.coords != self_coords:
            self_coo_ordered_dict = obj._reorder(obj.coo, obj.dims, obj.coords, self_dims)
            self_array = array(coo=self_coo_ordered_dict['coo'], dims=self_dims, coords=self_coords, dtype=obj.dtype, order=None)
            self_inv_dense = self_array.dense_.T
        else:
            self_inv_dense = obj.dense_.T
        return self_inv_dense

    def _pre_operation_with_array(self, other):
        assert self.order == other.order, "'order' attribute must be equal in both arrays"
        assert self.dtype == other.dtype, "'dtype' attribute must be equal in both arrays"
        commondims = self._union_dims(other, self.order)
        commoncoords = self._union_coords(other, commondims)
        self_inv_dense = self._get_inv_dense(self,commondims,commoncoords)
        other_inv_dense = self._get_inv_dense(other,commondims,commoncoords)
        return self_inv_dense, other_inv_dense, commondims, commoncoords

    def _pre_operation_with_number(self):
        return self.coo[:,-1]

    def _post_operation_with_array(self, resulting_dense, dims, coords):
        dense = resulting_dense.T
        dindex = self._dindex(coords)
        coo_dense = np.hstack([np.array(dindex, dtype=object), dense.reshape(dense.size,1)])
        coo_dense = coo_dense[coo_dense[:,-1] != 0]
        coo = coo_dense[coo_dense[:,-1] != np.nan]
        return array(coo=coo, dims=dims, coords=coords, dtype=self.dtype, order=self.order, dense=dense, dindex=dindex)

    def _post_operation_with_number(self, resulting_dense):
        coo_dense = np.hstack([self.coo[:,range(len(self.dims))], resulting_dense.reshape(resulting_dense.size,1)])
        coo_dense = coo_dense[coo_dense[:,-1] != 0]
        coo = coo_dense[coo_dense[:,-1] != np.nan]
        return array(coo=coo, dims=self.dims, coords=self.coords, dtype=self.dtype, order=self.order)

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

    def to_dataframe(self):
        mi = pd.MultiIndex.from_tuples(self.dindex_, names=self.dims)
        df = pd.DataFrame(pd.Series(self.dense_.reshape(-1), index=mi, name='value'))
        return df

    def to_arrow(self):
        '''
        Returns an Arrow table with custom metadata with a metadata key 'karray' that can be obtained by deserializing with json

        See also karray.from_feather(path)
        '''
        table = pa.Table.from_pandas(pd.DataFrame(self.coo))
        existing_meta = table.schema.metadata
        custom_meta_key = 'karray'
        custom_metadata = {}
        attr = ['dims', 'coords', 'dtype','order']
        for k,v in self._repo.items():
            if k in attr:
                custom_metadata[k] = v
        custom_meta_json = json.dumps(custom_metadata)
        existing_meta = table.schema.metadata
        combined_meta = {custom_meta_key.encode() : custom_meta_json.encode(),**existing_meta}
        return table.replace_schema_metadata(combined_meta)

    def to_feather(self, path):
        table = self.to_arrow()
        ft.write_feather(table, path)
        return None

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
        return array(coo=coo, dims=self.dims, coords=new_coords, dtype=self.dtype, order=self.order)

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
        return array(coo=coo, dims=self.dims, coords=coords, dtype=self.dtype, order=self.order)

    def _add_dim(self, dim:str, unique_coords:list):
        '''
        Add a new dimension to the array with only one element.

        Args:
            dim (str): the new dimension name.
            unique_coords (list): the new coordinates.

        Returns:
            a new coo array with the coordinates specified in the kwargs.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4],'y':[2,5]})
            >>> ar = ar.add_dim(dim='z', unique_coords=['a'])
        '''
        assert dim not in self.dims, "Dimension must not be in dims"
        assert isinstance(unique_coords, list), "Unique_coords must be a list"
        assert len(set(unique_coords)) == 1, "Unique_coords length must be 1"
        coo = np.hstack((np.empty([self.coo.shape[0],1], self.coo.dtype), self.coo))
        coo[:,0] = unique_coords[0]
        dims = [dim] + self.dims
        coords = {}
        coords[dim] = unique_coords
        for k,v in self.coords.items():
            coords[k] = v
        return dict(coo=coo, dims=dims, coords=coords, dtype=self.dtype, order=self.order)

    def add_dim(self, dim, unique_coords):
        return array(**self._add_dim(dim=dim, unique_coords=unique_coords))

    def str2int(self, dim:str=None, use_index:bool=False):
        '''
        Convert the coordinates of a dimension to integers if the coordinates are numbered strings.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':['a01','a02'],'y':['c','d']})
            >>> ar = ar.str2int(dim='x')
        '''
        assert dim in self.dims, "Dimension must be in dims"
        if use_index:
            int_coords = list(range(len(self.coords[dim])))
        else:
            int_coords = [int(''.join(filter(str.isdigit, elem))) for elem in self.coords[dim]]
            assert len(set(int_coords)) == len(int_coords), "There are duplicate values. You can use use_index=True to use the index instead of getting the numbers from string."
        return self.coords_replace(dim=dim, unique_coords=int_coords)


    ########## Methods for converting arrays to all integers ##########

    @staticmethod
    def _int_coords(obj, commondims:list, commoncoords:dict):
        scoo = obj.coo.copy()

        int_coords = {}
        for dim in commondims:
            int_dim_coords = list(range(len(commoncoords[dim])))
            int_coords[dim] = int_dim_coords
            if dim in obj.dims:
                scoo[:,obj.dims.index(dim)] = np.array(int_dim_coords)[np.array(commoncoords[dim]).searchsorted(scoo[:,obj.dims.index(dim)], sorter=int_dim_coords)]
        return scoo, int_coords

    @staticmethod
    def _coords_replace(coo, dims:list, coords:dict):
        coo_ = coo.copy()
        for dim in dims:
            if dim in coords:
                dim_coords = coords[dim]
                coo_[:,dims.index(dim)] = np.array(dim_coords)[np.array(coords[dim]).searchsorted(coo_[:,dims.index(dim)], sorter=list(range(len(coords[dim]))))]
        return coo

    def all_int(self, obj, commondims, commoncoords):
        '''
        Returns an array with all coordinates integers.
        '''
        coo, _, int_coords = self._int_coords(obj, commondims=commondims, commoncoords=commoncoords)
        coo_ = self._coords_replace(coo=coo, dims=obj.dims, coords=int_coords)
        return array(coo=coo_, dims=obj.dims, coords=int_coords, dtype=obj.dtype, order=obj.order)



def concat(arrays:list):
    dims = arrays[0].dims[:]
    dtype = arrays[0].dtype
    order = arrays[0].order
    assert all([isinstance(arr, array) for arr in arrays]), "All list items must be karray.array"
    assert all([set(arr.dims) == set(dims) for arr in arrays]), "All array must have the same dimensions"
    assert all([arr.dtype == dtype for arr in arrays]), "All array must have the same dtype"
    coo = np.vstack([arr.coo for arr in arrays])
    coords = reduce(lambda x,y: array(coo=x.coo, dims=x.dims, coords=x._union_coords(y,x.dims), dtype=x.dtype, order=x.order), arrays).coords
    return array(coo=coo, dims=dims, coords=coords, dtype=arrays[0].dtype, order=arrays[0].order)


def from_pandas(df, dims: list=None, dtype:str='float64', order=None):
    #TODO Sort dataframe index first of all and verify the consistency with the resulted array
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
        df = df[df['value'] != 0]
        dt = df[df['value'] != np.nan].reset_index()
        col = dt.pop("value")
        dt = dt[dims]
        dt.insert(len(dt.columns), col.name, col)
        coo = dt.to_numpy()
    else:
        df = df[df['value'] != 0]
        dt = df[df['value'] != np.nan]
        col = dt.pop("value")
        dt = dt[dims]
        dt.insert(len(dt.columns), col.name, col)
        coo = dt.to_numpy()
    return array(coo=coo,dims=dims,coords=coords, dtype=dtype, order=order)

def from_feather(path, with_table=False):
    restored_table = ft.read_table(path)
    custom_meta_key = 'karray'
    restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
    restored_meta = json.loads(restored_meta_json)
    coo = restored_table.to_pandas().to_numpy()
    if with_table:
        return array(coo=coo, **restored_meta), restored_table
    return array(coo=coo, **restored_meta)


# @nb.njit(parallel=True)
# def in1d_vec_nb(matrix, index_to_remove):
#     #matrix and index_to_remove have to be numpy arrays
#     #if index_to_remove is a list with different dtypes this 
#     #function will fail
#     out=np.empty(matrix.shape[0],dtype=nb.boolean)
#     index_to_remove_set=set(index_to_remove)
#     for i in nb.prange(matrix.shape[0]):
#         if matrix[i] in index_to_remove_set:
#             out[i]=False
#         else:
#             out[i]=True
#     return out

# @nb.njit(parallel=True)
# def in1d_scal_nb(matrix, index_to_remove):
#     #matrix and index_to_remove have to be numpy arrays
#     #if index_to_remove is a list with different dtypes this 
#     #function will fail
#     out=np.empty(matrix.shape[0],dtype=nb.boolean)
#     for i in nb.prange(matrix.shape[0]):
#         if (matrix[i] == index_to_remove):
#             out[i]=False
#         else:
#             out[i]=True
#     return out

# def isin_nb(matrix_in, index_to_remove):
#     #both matrix_in and index_to_remove have to be a np.ndarray
#     #even if index_to_remove is actually a single number
#     shape=matrix_in.shape
#     if index_to_remove.shape==():
#         res=in1d_scal_nb(matrix_in.reshape(-1),index_to_remove.take(0))
#     else:
#         res=in1d_vec_nb(matrix_in.reshape(-1),index_to_remove)
#     return res.reshape(shape)
