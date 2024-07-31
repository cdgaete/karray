import unittest
import numpy as np
from numpy.testing import assert_array_equal
from karray import Array, settings


class TestArrayInsert(unittest.TestCase):
    def setUp(self):
        value = [1.0, 2.0, 3.0]
        index = {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
        data = (index, value)
        coords = {'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        self.array_obj = Array(data=data, coords=coords)
    
    def test_insert_new_dim_int(self):
        new_data = {'z': 99}
        new_array = self.array_obj.insert(**new_data)
        expected_coords = { 'z': np.array([99], dtype=np.int64), 'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        assert_array_equal(new_array.coords['z'], expected_coords['z'])
        self.assertEqual(new_array.dims, ['z', 'x', 'y'])

    def test_insert_new_dim_float(self):
        new_data = {'z': 5.0}
        new_array = self.array_obj.insert(**new_data)
        expected_coords = {'z': np.array([5.0], dtype=np.float64), 'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        assert_array_equal(new_array.coords['z'], expected_coords['z'])
        self.assertEqual(new_array.dims, ['z', 'x', 'y'])
    
    def test_insert_new_dim_str(self):
        new_data = {'z': 'kkk'}
        new_array = self.array_obj.insert(**new_data)
        expected_coords = {'z': np.array(['kkk'], dtype=np.object_), 'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        assert_array_equal(new_array.coords['z'], expected_coords['z'])
        self.assertEqual(new_array.dims, ['z', 'x', 'y'])
    
    def test_insert_existing_dim_dict_map(self):
        new_data = {'z': {'y': {'a': 1, 'b': 2, 'c': 3}}}
        new_array = self.array_obj.insert(**new_data)
        expected_coords = {'z': np.array([1, 2, 3], dtype=np.int64), 'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        assert_array_equal(new_array.coords['z'], expected_coords['z'])
        self.assertEqual(new_array.dims, ['z', 'x', 'y'])
    
    def test_insert_existing_dim_list(self):
        new_data = {'z': {'y': [['a', 'b', 'c'], [1, 2, 3]]}}
        new_array = self.array_obj.insert(**new_data)
        expected_coords = {'z': np.array([1, 2, 3], dtype=np.int64), 'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        assert_array_equal(new_array.coords['z'], expected_coords['z'])
        self.assertEqual(new_array.dims, ['z', 'x', 'y'])
    
    def test_insert_existing_dim_duplicate(self):
        new_data = {'z': {'y': {'a': 2, 'b': 2, 'c': 2}}}
        new_array = self.array_obj.insert(**new_data)
        expected_coords = {'z': np.array([2], dtype=np.int64), 'x': np.array([1, 2, 3]), 'y': np.array(['a', 'b', 'c'])}
        assert_array_equal(new_array.coords['z'], expected_coords['z'])
        self.assertEqual(new_array.dims, ['z', 'x', 'y'])
    
    def test_insert_partial_mapping_error(self):
        with self.assertRaises(AssertionError):
            new_data = {'z': {'y': {'b': 2}}}
            self.array_obj.insert(**new_data)
    
    def test_insert_ndarray_invalid_type_error(self):
        with self.assertRaises(AssertionError):
            new_data = {'z': np.array([4, 5, 6])}
            self.array_obj.insert(**new_data)

if __name__ == '__main__':
    settings.data_obj = 'sparse'
    unittest.main()