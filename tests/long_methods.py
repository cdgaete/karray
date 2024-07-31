import unittest
import numpy as np
from numpy.testing import assert_array_equal
from karray import Long


class TestLong(unittest.TestCase):
    def setUp(self):
        # create Long objects for testing
        self.long_obj_with_str = Long(index={'holiday': ['aaa', 'bbb', 'ccc']}, value=[4.0, 5.0, 6.0])
        self.long_obj_with_int = Long(index={'holiday': [1, 2, 3, 5]}, value=[4.0, 5.0, 6.0, 7.0])
        self.long_obj_with_dt = Long(index={'holiday': np.array(
            ['2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]')}, value=[4.0, 5.0, 6.0])
        self.long_obj_with_list_dt = Long(index={'holiday': list(
            np.array(['2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]'))}, value=[4.0, 5.0, 6.0])
        self.long_obj_with_2d_dt = Long(index={'holiday': np.array(['2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]'), 'travel': np.array([
                                        '2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]')}, value=[4.0, 5.0, 6.0])

    def test_get_item_datetime64(self):
        actual = self.long_obj_with_dt['holiday', list(np.array(['2022-11-23'], dtype='datetime64[ns]'))]
        expected = Long({'holiday': np.array(['2022-11-23'], dtype='datetime64[ns]')}, [6.0])
        self.assertEqual(expected, actual)

    def test_get_item_ndarray(self):
        actual = self.long_obj_with_dt['holiday', np.array(['2022-11-23'], dtype='datetime64[ns]')]
        expected = Long({'holiday': np.array(['2022-11-23'], dtype='datetime64[ns]')}, [6.0])
        self.assertEqual(expected, actual)

    def test_get_item_list_string(self):
        actual = self.long_obj_with_str['holiday', ['ccc']]
        expected = Long({'holiday': ['ccc']}, [6.0])
        self.assertEqual(expected, actual)

    def test_get_item_list_int(self):
        actual = self.long_obj_with_int['holiday', [1, 3]]
        expected = Long({'holiday': [1, 3]}, [4.0, 6.0])
        self.assertEqual(expected, actual)

    def test_get_item_list_slice(self):
        actual = self.long_obj_with_int['holiday', 3:]
        expected = Long({'holiday': [3, 5]}, [6.0, 7.0])
        self.assertEqual(expected, actual)

    def test_index(self):
        actual = self.long_obj_with_dt.index
        expected = {'holiday': np.array(['2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]')}
        self.assertTrue(all(np.array_equal(act, exp) for act, exp in zip(actual, expected)))

    def test_value(self):
        actual = self.long_obj_with_dt.value
        expected = np.array([4.0, 5.0, 6.0])
        assert_array_equal(expected, actual)

    def test_dims(self):
        actual = self.long_obj_with_str.dims
        expected = ['holiday']
        self.assertEqual(expected, actual)

    def test_size(self):
        actual = self.long_obj_with_dt.size
        expected = 3
        self.assertEqual(expected, actual)

    def test_ndim(self):
        actual = self.long_obj_with_str.ndim
        expected = 1
        self.assertEqual(expected, actual)

    def test_insert_dict_and_list(self):
        actual = self.long_obj_with_list_dt.insert(travel={'holiday': [np.array(
            ['2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]'), np.array(['2022-11-16', '2022-11-17', '2022-11-18'], dtype='datetime64[ns]')]})
        expected = Long({'travel': list(np.array(['2022-11-16', '2022-11-17', '2022-11-18'], dtype='datetime64[ns]')), 'holiday': list(
            np.array(['2022-11-25', '2022-11-24', '2022-11-23'], dtype='datetime64[ns]'))}, [4.0, 5.0, 6.0])
        self.assertEqual(expected, actual)

    def test_rename(self):
        actual = self.long_obj_with_dt.rename(holiday='travel')
        expected = Long({'travel': np.array(['2022-11-25', '2022-11-24',
                        '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        self.assertEqual(expected, actual)

    def test_drop(self):
        actual = self.long_obj_with_2d_dt.drop('holiday')
        expected = Long({'travel': np.array(['2022-11-25', '2022-11-24',
                        '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        self.assertEqual(expected, actual)

    def test_items(self):
        actual = list(self.long_obj_with_dt.items())
        expected = [('holiday', np.array(['2022-11-25', '2022-11-24', '2022-11-23'],
                     dtype='datetime64[ns]')), ('value', np.array([4.0, 5.0, 6.0]))]
        self.assertTrue(all(act[0] == exp[0] for act, exp in zip(actual, expected)))
        self.assertTrue(all(np.array_equal(act[1], exp[1]) for act, exp in zip(actual, expected)))

    def test_eq_long(self):
        A_long = Long({'holiday': np.array(['2022-11-25', '2022-11-24',
                      '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        B_long = Long({'holiday': np.array(['2022-11-25', '2022-11-24',
                      '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        self.assertEqual(A_long, B_long)

    def test_ne_long_by_dim(self):
        A_long = Long({'holiday': np.array(['2022-11-25', '2022-11-24',
                      '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        B_long = Long({'travel': np.array(['2022-11-25', '2022-11-24', '2022-11-23'],
                      dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        self.assertNotEqual(A_long, B_long)

    def test_ne_long_by_dim_value(self):
        A_long = Long({'holiday': np.array(['2022-11-25', '2022-11-24',
                      '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        B_long = Long({'holiday': np.array(['2022-11-27', '2022-11-27',
                      '2022-11-27'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        self.assertNotEqual(A_long, B_long)

    def test_ne_long_by_value(self):
        A_long = Long({'holiday': np.array(['2022-11-25', '2022-11-24',
                      '2022-11-23'], dtype='datetime64[ns]')}, [4.0, 5.0, 6.0])
        B_long = Long({'holiday': np.array(['2022-11-25', '2022-11-24',
                      '2022-11-23'], dtype='datetime64[ns]')}, [1.0, 1.0, 1.0])
        self.assertNotEqual(A_long, B_long)

    def test_eq_value_float(self):
        actual = self.long_obj_with_dt == 5.0
        expected = np.array([False, True, False])
        assert_array_equal(expected, actual)

    def test_eq_value_int(self):
        actual = self.long_obj_with_dt == 5
        expected = np.array([False, True, False])
        assert_array_equal(expected, actual)

    def test_eq_value_nan(self):
        long = Long({'holiday': np.array(['2022-11-25', '2022-11-24', '2022-11-23'],
                    dtype='datetime64[ns]')}, [4.0, 5.0, np.nan])
        actual = long == np.nan
        expected = np.array([False, False, True])
        assert_array_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
