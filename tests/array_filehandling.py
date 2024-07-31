import unittest
import os
import numpy as np
from karray import Array, Long, from_pandas, from_polars, from_feather, from_csv, settings


class TestArrayFileHandling(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join(os.getcwd(), 'tests', 'data')
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.long = Long(index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10, 20])
        self.coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        self.arr = Array(data=self.long, coords=self.coords)

    def test_to_pandas_sparse(self):
        df = self.arr.to_pandas(dense=False)
        self.assertEqual(df.shape, (2, 3))
        self.assertEqual(list(df.columns), ['dim1', 'dim2', 'value'])
        self.assertTrue(np.array_equal(df['value'].values, np.array([10, 20])))

    def test_to_polars_sparse(self):
        df = self.arr.to_polars(dense=False)
        self.assertEqual(df.shape, (2, 3))
        self.assertEqual(list(df.columns), ['dim1', 'dim2', 'value'])
        self.assertTrue(np.array_equal(df['value'].to_numpy(), np.array([10, 20])))

    def test_to_pandas_dense(self):
        df = self.arr.to_pandas(dense=True)
        self.assertEqual(df.shape, (4, 3))
        self.assertEqual(list(df.columns), ['dim1', 'dim2', 'value'])
        self.assertTrue(np.array_equal(df['value'].values, np.array([10, 0, 0, 20])))

    def test_to_polars_dense(self):
        df = self.arr.to_polars(dense=True)
        self.assertEqual(df.shape, (4, 3))
        self.assertEqual(list(df.columns), ['dim1', 'dim2', 'value'])
        self.assertTrue(np.array_equal(df['value'].to_numpy(), np.array([10, 0, 0, 20])))

    def test_to_feather(self):
        self.arr.to_feather(os.path.join(self.data_dir, 'array.feather'))
        loaded_arr = from_feather(os.path.join(self.data_dir, 'array.feather'))
        self.assertEqual(loaded_arr.dims, self.arr.dims)
        self.assertEqual(loaded_arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(loaded_arr.todense(), self.arr.todense()))

    def test_from_pandas(self):
        df = self.arr.to_pandas(dense=False)
        arr = from_pandas(df, coords=self.coords)
        self.assertEqual(arr.dims, self.arr.dims)
        self.assertEqual(arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(arr.todense(), self.arr.todense()))

    def test_from_polars(self):
        df = self.arr.to_polars(dense=False)
        arr = from_polars(df, coords=self.coords)
        self.assertEqual(arr.dims, self.arr.dims)
        self.assertEqual(arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(arr.todense(), self.arr.todense()))

    def test_from_feather(self):
        self.arr.to_feather(os.path.join(self.data_dir, 'array.feather'))
        arr = from_feather(os.path.join(self.data_dir, 'array.feather'))
        self.assertEqual(arr.dims, self.arr.dims)
        self.assertEqual(arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(arr.todense(), self.arr.todense()))

    def test_from_csv(self):
        self.arr.to_pandas(dense=False).to_csv(os.path.join(self.data_dir, 'array.csv'), index=False)
        arr = from_csv(os.path.join(self.data_dir, 'array.csv'), coords=self.coords, with_='csv')
        self.assertEqual(arr.dims, self.arr.dims)
        self.assertEqual(arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(arr.todense(), self.arr.todense()))


if __name__ == '__main__':
    settings.data_type = 'sparse'
    unittest.main()
