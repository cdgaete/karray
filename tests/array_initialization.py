import unittest
import numpy as np
from karray import Array, settings

class TestArrayInitialization(unittest.TestCase):
    def setUp(self):
        self.long = ({'dim1': ['a', 'b'], 'dim2': [1, 2]}, [10, 20])
        self.coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        self.arr = Array(data=self.long, coords=self.coords)
    
    def test_init_from_long(self):
        arr = Array(data=self.long, coords=self.coords)
        self.assertEqual(arr.dims, ['dim1', 'dim2'])
        self.assertEqual(arr.shape, [2, 2])
        self.assertTrue(np.array_equal(arr.todense(), np.array([[10, 0], [0, 20]])))
    
    def test_init_from_tuple(self):
        index = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        value = [10, 20]
        arr = Array(data=(index, value), coords=self.coords)
        self.assertEqual(arr.dims, ['dim1', 'dim2'])
        self.assertEqual(arr.shape, [2, 2])
        self.assertTrue(np.array_equal(arr.todense(), np.array([[10, 0], [0, 20]])))
    
    def test_init_from_dense(self):
        dense = np.array([[10, 0], [0, 20]])
        arr = Array(data=dense, coords=self.coords)
        self.assertEqual(arr.dims, ['dim1', 'dim2'])
        self.assertEqual(arr.shape, [2, 2])
        self.assertTrue(np.array_equal(arr.todense(), dense))

    def test_array_with_invalid_data_type(self):
        with self.assertRaises(AssertionError):
            arr = Array(data=1, coords={'dim1': ['a', 'b'], 'dim2': [1, 2]})

    def test_init_from_sparse(self):
        try:
            import sparse as sp
        except ImportError:
            self.skipTest('sparse not installed')
        sparse_arr = sp.COO(data=[10, 20], coords=[[0, 1], [0, 1]], shape=(2, 2))
        arr = Array(data=sparse_arr, coords=self.coords)
        self.assertEqual(arr.dims, ['dim1', 'dim2'])
        self.assertEqual(arr.shape, [2, 2])
        self.assertTrue(np.array_equal(arr.todense(), np.array([[10, 0], [0, 20]])))

if __name__ == '__main__':
    settings.data_obj = 'sparse'
    unittest.main()
