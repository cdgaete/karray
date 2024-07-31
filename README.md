
# \[k\]array: labeled multi-dimensional arrays

[![karray Status Badge](https://img.shields.io/pypi/v/karray.svg)](https://pypi.org/project/karray/)
[![karray Python Versions](https://img.shields.io/pypi/pyversions/karray.svg)](https://pypi.org/project/karray/)
[![karray license](https://img.shields.io/pypi/l/karray.svg)](https://pypi.org/project/karray/)
[![Downloads](https://static.pepy.tech/badge/karray)](https://pepy.tech/project/karray)
[![Pipeline](https://gitlab.com/diw-evu/karray/badges/main/pipeline.svg)](https://gitlab.com/diw-evu/karray/-/commits/main)

Karray is a simple tool that intends to abstract the users from the complexity of working with labelled multi-dimensional arrays. Numpy is the tool’s core, with an extensive collection of high-level mathematical functions to operate on multi-dimensional arrays efficiently thanks to its well-optimized C code. With Karray, we put effort into generating lightweight objects expecting to reduce overheads and avoid large loops that cause bottlenecks and impact performance. Numpy is the only relevant dependency, while Polars, Pandas, sparse and Pyarrow are required to import, export and store the arrays. `karray` is developed by the research group `Transformation of the Energy Economy` at [DIW Berlin](https://www.diw.de/en/diw_01.c.604205.en/energy__transportation__environment_department.html) (German Institute of Economic Research).

**Links**

* Documentation: https://diw-evu.gitlab.io/karray
* Source code: https://gitlab.com/diw-evu/karray
* PyPI releases: https://pypi.org/project/karray

**Table of contents**

* [Quick installation](#quick-installation)
* [Importing karray](#importing-karray)
* [Usage Examples](#usage-examples)
    
    * [Creating an Array](#creating-an-array)
    * [Accessing Array Elements](#accessing-array-elements)
    * [Array Operations](#array-operations)
    * [Saving and Loading Arrays](#saving-and-loading-arrays)
    * [Interoperability with Other Libraries](#interoperability-with-other-libraries)
    

Getting started
===============

Quick installation
------------------

To install karray, you can use pip:

`pip  install  karray` 

Importing karray
----------------

To start using karray, import the necessary classes and functions:

```python
import karray as ka

# then you can use ka.Array, ka.Long, and ka.settings
```

The `Array` class represents a labeled multidimensional array, while the `Long` class represents a labeled one-dimensional array. The `settings` object allows you to configure various options for karray.

Usage Examples
--------------

### Creating an Array

You can create an `Array` object in several ways:

1.  From a `Long` object and coordinates:

```python
import pandas as pd

index = {'dim1': ['a', 'b'],
         'dim2': [1, 2],
         'dim3': pd.to_datetime(['2020-01-01', '2020-01-02'], utc=True)}
value = [10., 20.]
long = ka.Long(index=index, value=value)

arr1 = ka.Array(data=long)
arr1
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 64 bytes |
| Data object type | dense |
| Data object size | 64 bytes |
| Dimensions | \['dim1', 'dim2', 'dim3'\] |
| Shape | \[2, 2, 2\] |
| Capacity | 8   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |
| **dim3** | 2   | datetime64\[ns\] | \['2020-01-01T00:00:00.000000000' '2020-01-02T00:00:00.000000000'\] |

Data

|     | dim1 | dim2 | dim3 | value |
| --- | --- | --- | --- | --- |
| **0** | a   | 1   | 2020-01-01T00:00:00.000000000 | 10.00 |
| **1** | b   | 2   | 2020-01-02T00:00:00.000000000 | 20.00 |

2.  From a tuple of index and value, and coordinates:

```python
index2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
value2 = [10, 20]
coords2 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}

arr2 = ka.Array(data=(index2, value2), coords=coords2)
arr2
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10  |
| **1** | b   | 2   | 20  |

3.  From a dense NumPy array and coordinates:

```python
import numpy as np
dense = np.array([[10, 20], [30, 40]])
coords3 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}

arr3 = ka.Array(data=dense, coords=coords3)
arr3
```


**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 96 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 4   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10.00 |
| **1** | a   | 2   | 20.00 |
| **2** | b   | 1   | 30.00 |
| **3** | b   | 2   | 40.00 |

4.  From a sparse array (using the `sparse` library) and coordinates:

```python
import sparse as sp

sparse_arr = sp.COO(data=[10, 20], coords=[[0, 1], [0, 1]], shape=(2, 2))
coords4 = {'dim1': ['a', 'b'], 'dim2': [1, 2]}

arr4 = ka.Array(data=sparse_arr, coords=coords4)
arr4
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10  |
| **1** | b   | 2   | 20  |

### Accessing Array Elements

You can access elements of an `Array` object using various methods:

1.  Using the `items()` method to iterate over the array elements:

```python
for item in arr3.items():
    print(item)
```

    ('dim1', array(['a', 'a', 'b', 'b'], dtype=object))
    ('dim2', array([1, 2, 1, 2]))
    ('value', array([10., 20., 30., 40.]))
    

2.  Using the `to_pandas()` method to convert the array to a pandas DataFrame:

```python
df = arr1.to_pandas()
print(df)
```

      dim1  dim2       dim3  value
    0    a     1 2020-01-01   10.0
    1    b     2 2020-01-02   20.0
    

3.  Using the `to_polars()` method to convert the array to a polars DataFrame:

```python
df = arr1.to_polars()
print(df)
```

    shape: (2, 4)
    ┌──────┬──────┬─────────────────────┬───────┐
    │ dim1 ┆ dim2 ┆ dim3                ┆ value │
    │ ---  ┆ ---  ┆ ---                 ┆ ---   │
    │ str  ┆ i64  ┆ datetime[ns]        ┆ f64   │
    ╞══════╪══════╪═════════════════════╪═══════╡
    │ a    ┆ 1    ┆ 2020-01-01 00:00:00 ┆ 10.0  │
    │ b    ┆ 2    ┆ 2020-01-02 00:00:00 ┆ 20.0  │
    └──────┴──────┴─────────────────────┴───────┘
    

### Array Operations

karray provides various operations that can be performed on `Array` objects:

1.  Arithmetic operations:

```python
result = arr1 + arr2
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 128 bytes |
| Data object type | dense |
| Data object size | 64 bytes |
| Dimensions | \['dim1', 'dim2', 'dim3'\] |
| Shape | \[2, 2, 2\] |
| Capacity | 8   |
| Rows | 4   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |
| **dim3** | 2   | datetime64\[ns\] | \['2020-01-01T00:00:00.000000000' '2020-01-02T00:00:00.000000000'\] |

Data

|     | dim1 | dim2 | dim3 | value |
| --- | --- | --- | --- | --- |
| **0** | a   | 1   | 2020-01-01T00:00:00.000000000 | 20.00 |
| **1** | a   | 1   | 2020-01-02T00:00:00.000000000 | 10.00 |
| **2** | b   | 2   | 2020-01-01T00:00:00.000000000 | 20.00 |
| **3** | b   | 2   | 2020-01-02T00:00:00.000000000 | 40.00 |

```python
result = arr3 * 2
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 96 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 4   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 20.00 |
| **1** | a   | 2   | 40.00 |
| **2** | b   | 1   | 60.00 |
| **3** | b   | 2   | 80.00 |

```python
result = arr4 - 1
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 96 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 4   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 9.00 |
| **1** | a   | 2   | -1.00 |
| **2** | b   | 1   | -1.00 |
| **3** | b   | 2   | 19.00 |

2.  Comparison operations:

```python
mask = arr2 &gt; 10

mask = arr2 == 5
```

3.  Logical operations:

```python
result = arr2 &amp; arr4

result = arr2 | arr4

result = ~arr2
```

4.  Reduction operations:

```python
result = arr1.reduce('dim1', aggfunc='sum')
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim2', 'dim3'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim2** | 2   | int64 | \[1 2\] |
| **dim3** | 2   | datetime64\[ns\] | \['2020-01-01T00:00:00.000000000' '2020-01-02T00:00:00.000000000'\] |

Data

|     | dim2 | dim3 | value |
| --- | --- | --- | --- |
| **0** | 1   | 2020-01-01T00:00:00.000000000 | 10.00 |
| **1** | 2   | 2020-01-02T00:00:00.000000000 | 20.00 |

```python
result = arr1.reduce('dim2', aggfunc=np.mean)
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim3'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim3** | 2   | datetime64\[ns\] | \['2020-01-01T00:00:00.000000000' '2020-01-02T00:00:00.000000000'\] |

Data

|     | dim1 | dim3 | value |
| --- | --- | --- | --- |
| **0** | a   | 2020-01-01T00:00:00.000000000 | 5.00 |
| **1** | b   | 2020-01-02T00:00:00.000000000 | 10.00 |

5.  Shifting and rolling operations:

```python
shifted = arr3.shift(dim1=1, dim2=-1, fill_value=0.)
shifted
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 24 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 1   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | b   | 1   | 20.00 |

```python
rolled = arr3.roll(dim1=2)
rolled
```


**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 96 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 4   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10.00 |
| **1** | a   | 2   | 20.00 |
| **2** | b   | 1   | 30.00 |
| **3** | b   | 2   | 40.00 |

6.  Inserting new dimensions:

```python
# One dimension with one element
result = arr2.insert(dim3='x')
result
```


**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 64 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim3', 'dim1', 'dim2'\] |
| Shape | \[1, 2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim3** | 1   | object | \['x'\] |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim3 | dim1 | dim2 | value |
| --- | --- | --- | --- | --- |
| **0** | x   | a   | 1   | 10  |
| **1** | x   | b   | 2   | 20  |

```python
# One dimension with several elements related to an existing dimension using a dict
result = arr2.insert(dim3={'dim1': {'a': -1, 'b': -2}})
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 64 bytes |
| Data object type | dense |
| Data object size | 64 bytes |
| Dimensions | \['dim3', 'dim1', 'dim2'\] |
| Shape | \[2, 2, 2\] |
| Capacity | 8   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim3** | 2   | int64 | \[-2 -1\] |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim3 | dim1 | dim2 | value |
| --- | --- | --- | --- | --- |
| **0** | -1  | a   | 1   | 10  |
| **1** | -2  | b   | 2   | 20  |

```python
# One dimension with several elements related to an existing dimension using two lists
result = arr2.insert(dim3={'dim1': [['a', 'b'], [-1, -2]]})
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 64 bytes |
| Data object type | dense |
| Data object size | 64 bytes |
| Dimensions | \['dim3', 'dim1', 'dim2'\] |
| Shape | \[2, 2, 2\] |
| Capacity | 8   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim3** | 2   | int64 | \[-1 -2\] |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim3 | dim1 | dim2 | value |
| --- | --- | --- | --- | --- |
| **0** | -1  | a   | 1   | 10  |
| **1** | -2  | b   | 2   | 20  |

7.  Drop a dimension:

```python
result = arr1.drop('dim3')
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10.00 |
| **1** | b   | 2   | 20.00 |

!Note

    Dropping a dimension will work only if the resulting array still has unique coordinates. If dropping a dimension leads to an array with duplicate coordinates, as a results of the removed dimension, karray will raise an error.

```python
# Assertion error due to duplicate coords
try:
    arr3.drop('dim2')
except AssertionError as e:
    print(e)
```

    Index items per row must be unique. By removing ['dim2'] leads the existence of repeated indexes 
    e.g.:
      ('dim1',) value
    0 ('a',) 10.0
    1 ('a',) 20.0
    Intead, you can use obj.reduce('dim2')
    With an aggfunc: sum() by default
    

8.  Expanding a dimension (Broadcasting)

```python
result = arr3.expand(dim3=['x', 'y', 'z'])
result
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 384 bytes |
| Data object type | dense |
| Data object size | 96 bytes |
| Dimensions | \['dim1', 'dim2', 'dim3'\] |
| Shape | \[2, 2, 3\] |
| Capacity | 12  |
| Rows | 12  |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |
| **dim3** | 3   | object | \['x' 'y' 'z'\] |

Data

|     | dim1 | dim2 | dim3 | value |
| --- | --- | --- | --- | --- |
| **0** | a   | 1   | x   | 10.00 |
| **1** | a   | 1   | y   | 10.00 |
| **2** | a   | 1   | z   | 10.00 |
| **3** | a   | 2   | x   | 20.00 |
| **4** | a   | 2   | y   | 20.00 |
| **5** | a   | 2   | z   | 20.00 |
| **6** | b   | 1   | x   | 30.00 |
| **7** | b   | 1   | y   | 30.00 |
| **8** | b   | 1   | z   | 30.00 |
| **9** | b   | 2   | x   | 40.00 |
| **10** | b   | 2   | y   | 40.00 |
| **11** | b   | 2   | z   | 40.00 |

9.  ufunc operations

```python
arr3.ufunc(dim='dim2', func=np.prod, keepdims=True)
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 96 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 4   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 200.00 |
| **1** | a   | 2   | 200.00 |
| **2** | b   | 1   | 1200.00 |
| **3** | b   | 2   | 1200.00 |

!Note

    The dim argument is passed to ufunc as axis argument in numpy and keepdims argument is passed with the same name. You can add more arguments depending on the ufunc.

### Saving and Loading Arrays

karray supports saving and loading arrays using the Feather format:

1.  Saving an array to a Feather file:

```python
arr1.to_feather('array.feather')
```

2.  Loading an array from a Feather file:

```python
loaded_arr1 = ka.from_feather('array.feather')
loaded_arr1
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 64 bytes |
| Data object type | dense |
| Data object size | 64 bytes |
| Dimensions | \['dim1', 'dim2', 'dim3'\] |
| Shape | \[2, 2, 2\] |
| Capacity | 8   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |
| **dim3** | 2   | int64 | \[1577836800000000000 1577923200000000000\] |

Data

|     | dim1 | dim2 | dim3 | value |
| --- | --- | --- | --- | --- |
| **0** | a   | 1   | 2020-01-01T00:00:00.000000000 | 10.00 |
| **1** | b   | 2   | 2020-01-02T00:00:00.000000000 | 20.00 |

### Interoperability with Other Libraries

karray provides interoperability with other popular data manipulation libraries:

1.  Converting an array to a pandas DataFrame and then back to an array:

```python
df = arr2.to_pandas()
new_arr = ka.from_pandas(df, coords=coords2)
new_arr
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10  |
| **1** | b   | 2   | 20  |

2.  Converting an array to a polars DataFrame and then back to an array:

```python
df = arr2.to_polars()
new_arr = ka.from_polars(df, coords=coords2)
new_arr
```

**\[k\]array**

|     |     |
| --- | --- |
| Long object size | 48 bytes |
| Data object type | dense |
| Data object size | 32 bytes |
| Dimensions | \['dim1', 'dim2'\] |
| Shape | \[2, 2\] |
| Capacity | 4   |
| Rows | 2   |

Coords

| Dimension | Length | Type | Items |
| --- | --- | --- | --- |
| **dim1** | 2   | object | \['a' 'b'\] |
| **dim2** | 2   | int64 | \[1 2\] |

Data

|     | dim1 | dim2 | value |
| --- | --- | --- | --- |
| **0** | a   | 1   | 10  |
| **1** | b   | 2   | 20  |

There are many more features and functionalities. Please refer to the source code section for more details.

!Note

    karray is a work in progress. The API is subject to change in the future. We are looking for feedback, suggestions, and we appreciate your contributions.


© 2024 [Carlos Gaete-Morales](https://github.com/cdgaete)
