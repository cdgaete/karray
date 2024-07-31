import unittest
import numpy as np
from karray import Array, Long, settings


class TestArrayOperations(unittest.TestCase):
    def setUp(self):
        self.long = Long(
            index={'dim1': ['a', 'b'], 'dim2': [1, 2]}, value=[10., 20.])
        self.coords = {'dim1': ['a', 'b'], 'dim2': [1, 2]}
        self.arr = Array(data=self.long, coords=self.coords)

    def test_reduce(self):
        reduced_arr = self.arr.reduce('dim1', aggfunc='sum')
        self.assertEqual(reduced_arr.dims, ['dim2'])
        self.assertEqual(reduced_arr.shape, [2])
        self.assertTrue(np.array_equal(
            reduced_arr.todense(), np.array([10., 20.])))

    def test_shift(self):
        shifted_arr = self.arr.shift(dim1=1, fill_value=0.)
        self.assertEqual(shifted_arr.dims, self.arr.dims)
        self.assertEqual(shifted_arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(shifted_arr.todense(), np.array([[0., 0.], [10., 0.]])))

    def test_roll(self):
        rolled_arr = self.arr.roll(dim2=1)
        self.assertEqual(rolled_arr.dims, self.arr.dims)
        self.assertEqual(rolled_arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(rolled_arr.todense(),
                        np.array([[0., 10.], [20., 0.]])))

    def test_insert(self):
        inserted_arr = self.arr.insert(new_dim={'dim1': [['a', 'b'], ['d', 'c']]})
        self.assertEqual(inserted_arr.dims, ['new_dim', 'dim1', 'dim2'])
        self.assertEqual(inserted_arr.shape, [2, 2, 2])
        self.assertTrue(np.array_equal(inserted_arr.todense(),
                        np.array([[[10., 0.], [0., 0.]], [[0., 0.], [0., 20.]]])))

    def test_rename(self):
        renamed_arr = self.arr.rename(dim1='new_dim1')
        self.assertEqual(renamed_arr.dims, ['new_dim1', 'dim2'])
        self.assertEqual(renamed_arr.shape, self.arr.shape)
        self.assertTrue(np.array_equal(
            renamed_arr.todense(), self.arr.todense()))

    def test_drop(self):
        dropped_arr = self.arr.drop('dim1')
        self.assertEqual(dropped_arr.dims, ['dim2'])
        self.assertEqual(dropped_arr.shape, [2])
        self.assertTrue(np.array_equal(
            dropped_arr.todense(), np.array([10, 20])))

    def test_to_dict_sparse(self):
        items = self.arr.to_dict(dense=False)
        dims_and_value = [dim for dim, _ in items.items()]
        self.assertEqual(dims_and_value, ['dim1', 'dim2', 'value'])
        elems_and_value = [elem for _, elem in items.items()]
        expected_vectors = [np.array(['a', 'b']), np.array([1, 2]), np.array([10, 20])]
        for elem, expected_vector in zip(elems_and_value, expected_vectors):
            self.assertTrue(np.array_equal(elem, expected_vector))

    def test_to_dict_dense(self):
        items = self.arr.to_dict(dense=True)
        dims_and_value = [dim for dim, _ in items.items()]
        self.assertEqual(dims_and_value, ['dim1', 'dim2', 'value'])
        elems_and_value = [elem for _, elem in items.items()]
        expected_vectors = [np.array(['a', 'a', 'b', 'b']), np.array([1, 2, 1, 2]), np.array([10., 0., 0., 20.])]
        for elem, expected_vector in zip(elems_and_value, expected_vectors):
            self.assertTrue(np.array_equal(elem, expected_vector))


if __name__ == '__main__':
    settings.data_type = 'sparse'
    unittest.main()
