import unittest
import numpy as np
from karray import Array, settings


class TestKarray(unittest.TestCase):
    def test_mismatched_coordinates(self):
        # Create arrays with mismatched coordinates
        stock = Array(data=np.array([[10, 20, 0, 0],
                                     [0, 0, 300, 400]]),
                      coords={'origin': ['Canada', 'Brazil'],
                              'fruit': ['apple', 'orange', 'banana', 'mango']})

        price = Array(data=np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]),
                      coords={'origin': ['Canada', 'Brazil'],
                              'fruit': ['apple', 'banana', 'mango']})

        # Perform element-wise multiplication
        bill = stock * price

        # Define the expected result
        expected_result = Array(np.array([[1.0, 6.0, 0.0],
                                          [0.0, 90.0, 120.0]]))

        # Assert that the computed result matches the expected result
        self.assertTrue(bill == expected_result)

    def test_mismatched_coordinates_multiple_dims(self):
        # Create arrays with mismatched coordinates along multiple dimensions
        stock = Array(data=np.array([[10, 20, 0, 0],
                                     [0, 0, 300, 400]]),
                      coords={'origin': ['Canada', 'Brazil'],
                              'fruit': ['apple', 'orange', 'banana', 'mango']})

        price = Array(data=np.array([[0.1, 0.2],
                                     [0.3, 0.4]]),
                      coords={'origin': ['Canada', 'Brazil'],
                              'fruit': ['apple', 'orange']})

        # Perform element-wise multiplication
        bill = stock * price

        # Define the expected result
        expected_result = Array(np.array([[1.0, 4.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0]]))

        # Assert that the computed result matches the expected result
        self.assertTrue(bill == expected_result)


if __name__ == '__main__':
    settings.data_obj = 'dense'
    unittest.main()
