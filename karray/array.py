
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as ft
import json
from functools import reduce
from itertools import product
from typing import Union
from .utils import _format_bytes


class Coo:
    def __init__(self, data:object):
        if isinstance(data, list):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = [np.array(ar.tolist()) for ar in data.T]
        elif isinstance(data, Coo):
            self.data = data.data


    def _repr_html_(self):
        coo_nbytes = _format_bytes(sum([arr.nbytes for arr in self.data]))
        shape = (self.data[0].shape[0], len(self.data))
        return ('<h3>Coo</h3>'
            '<table>'
            f'<tr><th>data</th><td>{coo_nbytes}</td></tr>'
            f'<tr><th>shape</th><td>{shape}</td></tr>'
            '</table>')

    def __getitem__(self, item:Union[int,slice,np.ndarray,tuple]):
        if isinstance(item, (int,slice,range,np.ndarray)):
            rows = item
            cols = slice(0, len(self.data), 1)
            data = []
            for col in range(cols.start,cols.stop,cols.step):
                data.append(self.data[col][rows].copy())
            return Coo(data)
        elif isinstance(item, tuple):
            rows = item[0]
            column = item[1]
            if isinstance(column, (slice,range)):
                start = column.start
                stop = column.stop or len(self.data)
                step = column.step or 1
                cols = slice(start,stop,step)
                data = []
                for col in range(cols.start,cols.stop,cols.step):
                    data.append(self.data[col][rows].copy())
                return Coo(data)
            elif isinstance(column, list):
                data = []
                for col in column:
                    data.append(self.data[col][rows].copy())
                return Coo(data)
            elif isinstance(column, int):
                return self.data[column][rows]
            else:
                raise Exception("Column type not supported")
        else:
            raise Exception("Slice type not supported")

    def __iter__(self):
        return self.data

    def __eq__(self, __o: object):
        if isinstance(__o, np.generic):
            if np.isnan(__o):
                return self.data[-1] == __o
            else:
                raise Exception("np.array not supported yet")
        elif isinstance(__o, (int,float)):
            return self.data[-1] == __o
        else:
            raise Exception(f"{type(__o)} not supported yet")
    def __ne__(self, __o: object):
        if isinstance(__o, np.generic):
            if np.isnan(__o):
                return self.data[-1] != __o
            else:
                raise Exception("np.array not supported yet")
        elif isinstance(__o, (int,float)):
            return self.data[-1] != __o
        else:
            raise Exception(f"{type(__o)} not supported yet")

    # def __add__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = self.data[-1].copy() + other
    #         return Coo(data)
    #     else:
    #         raise Exception("Not supported yet")

    # def __mul__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = self.data[-1].copy() * other
    #         return Coo(data)
    #     else:
    #         raise Exception("Not supported yet")

    # def __sub__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = self.data[-1].copy() - other
    #         return Coo(data)
    #     else:
    #         raise Exception("Not supported yet")

    # def __truediv__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = self.data[-1].copy() / other
    #         return Coo(data)
    #     else:
    #         raise Exception("Not supported yet")

    # def __radd__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = other + self.data[-1].copy()
    #         return Coo(data)

    # def __rmul__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = other * self.data[-1].copy()
    #         return Coo(data)

    # def __rsub__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = other - self.data[-1].copy()
    #         return Coo(data)

    # def __rtruediv__(self, other):
    #     if isinstance(other, (int, float)):
    #         data = self.data[:]
    #         data[-1] = other / self.data[-1].copy()
    #         return Coo(data)


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
        self.coo = None
        self.dims = None
        self.coords = None
        self.dtype = dtype
        self.order = order
        coo, dims, coords = self._check_input(coo, dims, coords)
        # coo, dims, coords = self._check_input(Coo(coo), dims, coords) # new
        self.dense_ = dense
        self.cindex_ = cindex
        self.dindex_ = dindex
        self._initial_dims_rearrange(coo, dims, coords)
        # Experiment
        self.cooo = Coo(self.coo)
        return None

    def __repr__(self):
        return f"Karray.array(coo, dims, coords)"

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
            raise AttributeError(name) # ipython requirement for repr_html
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

    def _add_coords(self, coo, dims):
        coords = {}
        for i, dim in enumerate(dims):
            coords[dim] = list(np.sort(np.unique(coo[:,i])))
        return coords

    def _check_input(self, coo, dims, coords):
        assert all([arg is not None for arg in [coo, dims, self.dtype]]), "coo, dims and dtype must be defined"
        assert isinstance(coo, np.ndarray), "coo must be a numpy array"
        # assert isinstance(coo, Coo), "coo must be a numpy array" # new
        assert isinstance(dims, list), "dims must be a list"
        if coords is None:
            print("coords is None. self.coords has been generated from coo and dims.")
            coords = self._add_coords(coo, dims)
        assert isinstance(coords, dict), "coords must be a dictionary"
        assert len(dims) == len(coords), "Length of dims and coords must be equal"
        assert all([all(np.sort(np.unique(arr)) == np.sort(arr)) for arr in coords.values()]), "Coordinates must be unique."
        for i in range(len(dims)):
            assert set(coo[:,range(len(dims))].T[i]).issubset(coords[dims[i]]), f"All elements of coo array must be included in coords for dim: {self.dims[i]}."
            # assert set(coo[:,i]).issubset(coords[dims[i]]), f"All elements of coo array must be included in coords for dim: {self.dims[i]}." # new
        return coo, dims, coords

    def _initial_dims_rearrange(self, coo, dims, coords):
        oredered_dims = self._order_with_preference(dims, self.order)
        ordered_dict = self._reorder(coo, dims, coords, oredered_dims)
        flag = False
        if dims != ordered_dict['dims']:
            flag = True
        if coords != ordered_dict['coords']:
            flag = True
        elif tuple(coords.keys()) != tuple(ordered_dict['coords'].keys()):
            flag = True
        self.coo = ordered_dict['coo']
        self.dims = ordered_dict['dims']
        self.coords = ordered_dict['coords']
        if flag:
            if self.dense_ is not None:
                print("array.__init__: coords reordered self.dense_ switched to None")
                self.dense_ = None
                self.cindex_ = None
                self.dindex_ = None
        return None

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
        # return list(zip(*self._cindex.data)) # new

    def dindex(self):
        '''
        Returns the index of dense array as a list of tuples.
        '''
        return self._dindex(self.coords)

    def _dindex(self, coords):
        '''
        Returns the index of dense array as a list of tuples.
        '''
        return sorted(product(*list(coords.values())))

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
        hshTbl = {ai: idx for idx, ai in enumerate(self.dindex_)}
        ix_zeros = [hshTbl[bi] for bi in self.cindex_]
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
        # coo = self_coo[:, dim_order] # new
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
        elif len(self.dims) == 0 or len(other.dims) == 0:
            for obj in [self,other]:
                if len(obj.dims) > 0:
                    dims = obj.dims
            return self._order_with_preference(dims, preferred_order)
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
                raise Exception(f"Dimension {dim} not found in either arrays")
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
            if tuple(obj.coords.keys()) == tuple(self_coords.keys()):
                self_inv_dense = obj.dense_.T
            else:
                self_coo_ordered_dict = obj._reorder(obj.coo, obj.dims, obj.coords, self_dims)
                self_array = array(coo=self_coo_ordered_dict['coo'], dims=self_dims, coords=self_coords, dtype=obj.dtype, order=None)
                self_inv_dense = self_array.dense_.T
        return self_inv_dense

    def _pre_operation_with_array(self, other):
        assert self.order == other.order, "'order' attribute must be equal in both arrays"
        assert self.dtype == other.dtype, "'dtype' attribute must be equal in both arrays"
        commondims = self._union_dims(other, self.order)
        commoncoords = self._union_coords(other, commondims)
        self_inv_dense = self._get_inv_dense(self, commondims, commoncoords)
        other_inv_dense = self._get_inv_dense(other, commondims, commoncoords)
        return self_inv_dense, other_inv_dense, commondims, commoncoords

    def _pre_operation_with_number(self):
        return self.coo[:,-1]

    def _post_operation_with_array(self, resulting_dense, dims, coords):
        dense = resulting_dense.T
        dindex = self._dindex(coords)
        coo_dense = np.hstack([np.array(dindex, dtype=object), dense.reshape(dense.size,1)])
        # coo_dense = Coo(*Coo(np.array(dindex, dtype=object)) + [dense.reshape(dense.size,1)]) # new
        coo_dense = coo_dense[coo_dense[:,-1] != 0]
        # coo_dense = coo_dense[coo_dense != 0] # new
        coo = coo_dense[coo_dense[:,-1] != np.nan]
        # coo = coo_dense[coo_dense != np.nan] # new
        return array(coo=coo, dims=dims, coords=coords, dtype=self.dtype, order=self.order, dense=dense, dindex=dindex)

    def _post_operation_with_number(self, resulting_dense):
        coo_dense = np.hstack([self.coo[:,range(len(self.dims))], resulting_dense.reshape(resulting_dense.size,1)])
        # coo_dense = Coo(*self.coo[:,range(len(self.dims))] + [resulting_dense.reshape(resulting_dense.size,1)]) # new
        coo_dense = coo_dense[coo_dense[:,-1] != 0]
        # coo_dense = coo_dense[coo_dense != 0] # new
        coo = coo_dense[coo_dense[:,-1] != np.nan]
        # coo = coo_dense[coo_dense != np.nan] # new
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
            # resulting_dense = self_dense / other_dense

            # Avoid zero division warning
            # Method 1
            # resulting_dense = np.divide(self_dense, other_dense, out=np.zeros(self_dense.shape, dtype=float), where=other_dense!=0)
            # Method 2
            # mask = other_dense != 0.0
            # resulting_dense = self_dense.copy()
            # np.divide(self_dense, other_dense, out=resulting_dense, where=mask)
            # Method 3
            resulting_dense = np.zeros_like(self_dense)
            np.divide(self_dense, other_dense, out=resulting_dense, where=~np.isclose(other_dense,np.zeros_like(other_dense)))
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
        # table = pa.Table.from_pandas(pd.DataFrame(*self.coo)) # new
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
        maximum = max(masksum)
        if maximum > 0:
            mask = masksum == max(masksum) # when criteria unmatch any value in coo then max is zero ended up all true and keep coo as it is
            coo = self.coo[mask]
        else:
            coo = self._make_zero_coo()
        return array(coo=coo, dims=self.dims, coords=new_coords, dtype=self.dtype, order=self.order)

    def add_elem(self, **kwargs):
        coords = {}
        for dim in kwargs:
            assert dim in self.dims, f'dim: {dim} must exist in self.dims: {self.dims}'
        for dim in self.coords:
            if dim in kwargs:
                coords[dim] = sorted(set(self.coords[dim] + kwargs[dim]))
            else:
                coords[dim] = self.coords[dim]
        return array(coo=self.coo, dims=self.dims, coords=coords, dtype=self.dtype, order=self.order)

    @staticmethod
    def _reduce(obj, dim:str, aggfunc:str):
        '''
        aggfunc in ['sum','mul','mean']
        '''
        assert dim in obj.dims, f"dim {dim} not in self.dims: {obj.dims}"

        if aggfunc == 'sum':
            dense = np.add.reduce(obj.dense_, axis=obj.dims.index(dim))
        elif aggfunc == 'mul':
            dense = np.multiply.reduce(obj.dense_, axis=obj.dims.index(dim))
        elif aggfunc == 'mean':
            dense = np.average(obj.dense_, axis=obj.dims.index(dim))

        dims = [d for d in obj.dims if d != dim]
        coords = {k:v for k,v in obj.coords.items() if k in dims}
        dindex = obj._dindex(coords)
        coo_dense = np.hstack([np.array(dindex, dtype=object), dense.reshape(dense.size,1)])
        # coo_dense = Coo(Coo(np.array(dindex, dtype=object)).data + [dense.reshape(dense.size,1)]) # new
        coo_dense = coo_dense[coo_dense[:,-1] != 0]
        # coo_dense = coo_dense[coo_dense != 0] # new
        coo = coo_dense[coo_dense[:,-1] != np.nan]
        # coo = coo_dense[coo_dense != np.nan] # new
        return dict(coo=coo, dims=dims, coords=coords, dtype=obj.dtype, order=obj.order, dense=dense, dindex=dindex)

    def reduce(self, dim, aggfunc='sum'):
        '''
        aggfunc in ['sum','mul','mean'], defult 'sum'
        '''
        return array(**self._reduce(self, dim, aggfunc))

    @staticmethod
    def _add_dim_single(obj, **kwargs):
        '''
        Add a new dimension to the array with only one element.
        This function add the new dimention in position 0.
        The order of the new dimensions will be adjusted when creating the new array (see _initial_dims_rearrange)

        Args:
            kwargs: {dim:unique_coords} with:
                dim (str): the new dimension name.
                unique_coords (str or int): the new coordinate.

        Returns:
            Dictionary with new coo, dims and coords

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4],'y':[2,5]})
            >>> ar = ar.add_dim(z='a')
        '''
        for dim in kwargs:
            assert dim not in obj.dims, f"Dimension '{dim}' must not be in dims {obj.dims}"
            assert isinstance(kwargs[dim], (str,int)), "Unique_coords must be a string or integer"

        coo_ = obj.coo.copy()
        # coo_ = Coo(obj.coo) # new
        dims_ = obj.dims[:]
        coords = {}
        for dim in kwargs:
            coo_ = np.hstack((np.empty([coo_.shape[0],1], coo_.dtype), coo_))
            # arr = np.empty(coo_[:,-1].shape[0], dtype=np.dtype(type(kwargs[dim]).__name__)) # new
            # arr[:] = kwargs[dim] # new
            # coo_ = Coo([arr] + coo_.data) # new
            coo_[:,0] = kwargs[dim]
            # new
            dims_ = [dim] + dims_
            coords[dim] = [kwargs[dim]]
        for k,v in obj.coords.items():
            coords[k] = v
        return dict(coo=coo_, dims=dims_, coords=coords, dtype=obj.dtype, order=obj.order)

    @staticmethod
    def _add_dim_map(obj, replace=True, **kwargs):
        '''
        Add a new dimensions to the array with elements that match with an existing dimension elements.

        Args:
            kwargs: {dim:unique_coords} with:
            dim (str): the new dimension name.
            unique_coords (dict): dictionary with existing dims as keys and a nested dictionary that maps elements of an existing dimension with the new one.

        Returns:
            Dictionary with new coo, dims and coords.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4,6],'y':[2,5]})
            >>> ar = ar.add_dim(z={'x':{1:'m',4:'n',6:'m'}}) # Where 'm' and 'n' are the new elements, and match with each 'x' elements.
        '''
        for dim in kwargs:
            assert dim not in obj.dims, f"Dimension '{dim}' must not be in dims {obj.dims}"
            assert isinstance(kwargs[dim], dict), "Unique_coords must be a dictionary"
            assert len(kwargs[dim]) == 1, "Only one key is possible with an existing dim"
            existing_dim = list(kwargs[dim].keys())[0]
            assert existing_dim in obj.dims, f"The dictionary key {existing_dim} must be already in dims {obj.dims}"
            assert isinstance(kwargs[dim][existing_dim], dict), f"The dictionary value must be a nested dict that contains the elements map"

        new_dims = []
        pseudo_ordered_coords = {}

        dims_to_replace = {}
        for dim in kwargs:
            existing_dim = list(kwargs[dim].keys())[0]
            dims_to_replace[existing_dim] = dim
            new_dims.append(dim)
            pseudo_ordered = []
            for elem in obj.coords[existing_dim]:
                pseudo_ordered.append(kwargs[dim][existing_dim][elem])
            pseudo_ordered_coords[dim] = pseudo_ordered
        if replace:
            attr_dict = obj._rename(obj, **dims_to_replace)
            attr_dict = obj._coords_replace(attr_dict['coo'], attr_dict['dims'], attr_dict['coords'], new_dims, pseudo_ordered_coords)
            return dict(coo=attr_dict['coo'], dims=attr_dict['dims'], coords=attr_dict['coords'], dtype=obj.dtype, order=obj.order)
        else:
            coords = {}
            dims = []
            coo = np.hstack((np.empty([obj.coo.shape[0],len(dims_to_replace)], obj.coo.dtype), obj.coo))
            # coo = obj.coo.data[:] # new
            i = 0
            for existing_dim in dims_to_replace:
                coords[dims_to_replace[existing_dim]] = obj.coords[existing_dim]
                dims.append(dims_to_replace[existing_dim])
                coo[:,i] = obj.coo[obj.dims.index(existing_dim)]
                # coo.insert(i,obj.coo[:,obj.dims.index(existing_dim)]) # new
                i += 1
            # coo = Coo(coo) # new
            for dim in obj.coords:
                coords[dim] = obj.coords[dim]
                dims.append(dim)
            attr_dict = obj._coords_replace(coo, dims, coords, new_dims, pseudo_ordered_coords)
            return dict(coo=attr_dict['coo'], dims=attr_dict['dims'], coords=attr_dict['coords'], dtype=obj.dtype, order=obj.order)


    def add_dim(self, replace=True, **kwargs):
        first_value = list(kwargs.values())[0]
        if isinstance(first_value, (str,int)):
            return array(**self._add_dim_single(self,**kwargs))
        elif isinstance(first_value, dict):
            return array(**self._add_dim_map(self,replace,**kwargs))

    @staticmethod
    def _rename(obj, **kwargs):
        '''
        rename dims

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4,6],'y':[2,5]})
            >>> ar = ar.rename(x='z')
        '''
        for olddim, newdim in kwargs.items():
            assert olddim in obj.dims, f"Dimension {olddim} must be in dims {obj.dims}"
            assert newdim not in obj.dims, f"Dimension {newdim} must not be in dims {obj.dims}"
        
        coords = {}
        for dim, elems in obj.coords.items():
            if dim in kwargs:
                coords[kwargs[dim]] = elems
            else:
                coords[dim] = elems
        dims = obj.dims[:]
        for dim in kwargs:
            dims[dims.index(dim)] = kwargs[dim]
        return dict(coo=obj.coo, dims=dims, coords=coords, dtype=obj.dtype, order=obj.order)

    def rename(self, include_dense:bool=False, **kwargs):
        '''
        rename dims
        '''
        dims_coords_dict = self._rename(self, **kwargs)
        if include_dense:
            return array(dense=self.dense_, dindex=self.dindex_,**dims_coords_dict)
        else:
            return array(**dims_coords_dict)

    #TODO: add elements to coords. Useful from the beginning. This will avoid always getting new dense when native symbols deal with derived ones.

    # def coords_replace(self, dim:str=None, unique_coords:list=None):
    #     '''
    #     Replace the coordinates of a dimension.

    #     Args:
    #         dim (str): the dimension to replace.
    #         unique_coords (list): the new coordinates.

    #     Returns:
    #         a new coo array with the coordinates specified in the kwargs.

    #     Example:
    #         >>> import karray as ka
    #         >>> import numpy as np
    #         >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':[1,4],'y':[2,5]})
    #         >>> ar = ar.coords_replace(dim='x', unique_coords=['a','b'])
    #     '''
    #     assert dim in self.dims, "Dimension must be in dims"
    #     assert isinstance(unique_coords, list), "Unique_coords must be a list"
    #     assert len(set(unique_coords)) == len(unique_coords), "Unique_coords must be unique"
    #     assert len(unique_coords) == len(self.coords[dim]), "Unique_coords must be the same length as the original coordinates"

    #     coo = self.coo.copy()
    #     coo[:,self.dims.index(dim)] = np.array(unique_coords)[np.array(self.coords[dim]).searchsorted(coo[:,self.dims.index(dim)], sorter=list(range(len(self.coords[dim]))))]
    #     coords = {k:v for k,v in self.coords.items()}
    #     coords[dim] = unique_coords
    #     return array(coo=coo, dims=self.dims, coords=coords, dtype=self.dtype, order=self.order)

    @staticmethod
    def _coords_replace(obj_coo, obj_dims, obj_coords, dims:list, pseudocoords:dict):
        '''
        dims a list that contains the dimensions to replace the coords
        pseudocoords contains the new dimensions coords, not necesarilly has to be unique, though.
        '''
        assert set(dims).issubset(set(pseudocoords)), "dims must be a subset of pseudocoords keys"
        assert set(dims).issubset(set(obj_dims))
        coo = obj_coo.copy()
        # coo = obj_coo.data[:] # new
        for dim in dims:
            if dim in pseudocoords:
                assert len(obj_coords[dim]) == len(pseudocoords[dim]), f"pseudocoords[{dim}] must have the same length as obj_coords[{dim}]"
                dim_coords = pseudocoords[dim]
                coo[:,obj_dims.index(dim)] = np.array(dim_coords)[np.array(obj_coords[dim]).searchsorted(obj_coo[:,obj_dims.index(dim)], sorter=list(range(len(obj_coords[dim]))))]
                # coo[obj_dims.index(dim)] = np.array(dim_coords)[np.array(obj_coords[dim]).searchsorted(obj_coo[:,obj_dims.index(dim)], sorter=list(range(len(obj_coords[dim]))))] # new
        coo = Coo(coo)
        coords = {}
        for dim in obj_coords:
            if dim in pseudocoords:
                coords[dim] = sorted(set(pseudocoords[dim]))
            else:
                coords[dim] = obj_coords[dim]
        return dict(coo=coo, dims=obj_dims, coords=coords)

    @staticmethod
    def _str2int(obj, dim:str=None, use_index:bool=False):
        '''
        Convert the coordinates of a dimension to integers if the coordinates are numbered strings.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':['a01','a02'],'y':['c','d']})
            >>> ar = ar.str2int(dim='x')
        '''
        assert dim in obj.dims, "Dimension must be in dims"
        if use_index:
            int_coords = list(range(len(obj.coords[dim])))
        else:
            int_coords = [int(''.join(filter(str.isdigit, elem))) for elem in obj.coords[dim]]
            assert len(set(int_coords)) == len(int_coords), "There are duplicate values. You can use use_index=True to use the index instead of getting the numbers from string."
        coo_coords_dict =  obj._coords_replace(obj.coo,obj.dims,obj.coords, dims=[dim], pseudocoords={dim:int_coords})
        return dict(coo=coo_coords_dict['coo'], dims=obj.dims, coords=coo_coords_dict['coords'], dtype=obj.dtype, order=obj.order)

    def str2int(self, dim:str=None, use_index:bool=False, include_dense:bool=False):
        '''
        Convert the coordinates of a dimension to integers if the coordinates are numbered strings.

        Example:
            >>> import karray as ka
            >>> import numpy as np
            >>> ar = ka.array(coo = np.array([[1,2,3],[4,5,6]]), dims = ['x','y'], coords = {'x':['a01','a02'],'y':['c','d']})
            >>> ar = ar.str2int(dim='x')
        '''
        array_dict = self._str2int(self, dim=dim, use_index=use_index)
        if include_dense:
            return array(dense=self.dense_, dindex=self.dindex_, **array_dict)
        else:
            return array(**array_dict)

    @staticmethod
    def _elems2datetime(obj, dim:str=None, reference_date:str='01-01-2030', freq:str='H'):
        assert dim in obj.dims
        len_coord = len(obj.coords[dim])
        start_date = pd.to_datetime(reference_date)
        drange = pd.date_range(start_date, periods=len_coord, freq=freq)
        coo_coords_dict =  obj._coords_replace(obj.coo,obj.dims,obj.coords, dims=[dim], pseudocoords={dim:drange.to_numpy()})
        return dict(coo=coo_coords_dict['coo'], dims=obj.dims, coords=coo_coords_dict['coords'], dtype=obj.dtype, order=obj.order)

    def elems2datetime(self, dim:str=None, reference_date:str='01-01-2030', freq:str='H'):
        array_dict = self._elems2datetime(self, dim=dim, reference_date=reference_date, freq=freq)
        return array(**array_dict)

    def _make_zero_coo(self):
        '''
        select the first element of each dim and set value as zero. The results is an numpy array with one row.
        '''
        coo = np.array([], dtype=object).reshape((0,len(self.dims)+1))
        # coo = Coo(np.array([], dtype=object).reshape((0,len(self.dims)+1))) # new
        return coo

    ########## Methods for converting arrays to all integers ##########

    # def _int_coords(self):
    #     coo = self.coo.copy()
    #     int_coords = {}
    #     for dim in self.dims:
    #         int_dim_coords = list(range(len(self.coords[dim])))
    #         int_coords[dim] = int_dim_coords
    #         coo[:, self.dims.index(dim)] = np.array(int_dim_coords)[np.array(self.coords[dim]).searchsorted(self.coo[:, self.dims.index(dim)], sorter=int_dim_coords)]
    #     return dict(coo=coo, coords=int_coords)

    # def all_int(self):
    #     '''
    #     Returns an array with all coordinates integers.
    #     '''
    #     coo_coords_dict = self._int_coords()
    #     return array(coo=coo_coords_dict['coo'], dims=self.dims, coords=coo_coords_dict['coords'], dtype=self.dtype, order=self.order)

    # def dense_new(self):
    #     values = self.coo[:,-1]
    #     hlistd = []
    #     for t in self.dindex_:
    #         hlistd.append(hash(t))
    #     hlistc = []
    #     for t in self.cindex_:
    #         hlistc.append(hash(t))

    #     dindex = np.array(hlistd)
    #     cindex = np.array(hlistc)

    #     dindex_arg = np.argsort(dindex)
    #     dindex_ = dindex[dindex_arg]
    #     cindex_arg = np.argsort(cindex)
    #     cindex_ = cindex[cindex_arg]
    #     ix = np.searchsorted(dindex_, cindex_)
    #     ixa = dindex_arg[ix]
    #     zeros = np.zeros((len(self.dindex_),), dtype='float64')
    #     zeros[ixa] = values[cindex_arg]
    #     return zeros.reshape(self.shape)



def concat(arrays:list):
    dims = arrays[0].dims[:]
    dtype = arrays[0].dtype
    order = arrays[0].order
    assert all([isinstance(arr, array) for arr in arrays]), "All list items must be karray.array"
    assert all([set(arr.dims) == set(dims) for arr in arrays]), "All array must have the same dimensions"
    assert all([arr.dtype == dtype for arr in arrays]), "All array must have the same dtype"
    coo = np.vstack([arr.coo for arr in arrays])
    # coo = Coo([np.vstack([arr.coo.data[i] for arr in arrays]).reshape(-1) for i in range(len(dims) + 1)]) # new
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
        # coo = Coo(dt.to_numpy()) # new
    else:
        df = df[df['value'] != 0]
        dt = df[df['value'] != np.nan]
        col = dt.pop("value")
        dt = dt[dims]
        dt.insert(len(dt.columns), col.name, col)
        coo = dt.to_numpy()
        # coo = Coo(dt.to_numpy()) # new
    return array(coo=coo,dims=dims,coords=coords, dtype=dtype, order=order)

def from_feather(path, with_table=False):
    restored_table = ft.read_table(path)
    custom_meta_key = 'karray'
    restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
    restored_meta = json.loads(restored_meta_json)
    coo = restored_table.to_pandas().to_numpy()
    # coo = Coo(restored_table.to_pandas().to_numpy()) # new
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
