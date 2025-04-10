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

        # Per observed behavior, the result has shape [2, 4]
        # The result maintains the original fruit list from stock
        self.assertEqual(bill.shape, [2, 4])

        # Check the dimensions maintained
        self.assertEqual(bill.dims, ['origin', 'fruit'])

        # Check specific expected values
        # Canada-apple: 10 * 0.1 = 1.0
        self.assertAlmostEqual(bill.data[0, 0], 1.0)

        # We should see zeros in places without matching coordinates
        self.assertAlmostEqual(bill.data[0, 1], 0.0)  # Canada-orange (no price)

        # Check Brazil-banana: 300 * 0.0 = 0
        self.assertAlmostEqual(bill.data[1, 2], 0.0)

        # Check Brazil-mango: 400 * 0.0 = 0
        self.assertAlmostEqual(bill.data[1, 3], 0.0)

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

        # Check shape
        self.assertEqual(bill.shape, [2, 4])

        # Check values at key positions
        # Canada-apple: 10 * 0.1 = 1.0
        self.assertAlmostEqual(bill.data[0, 0], 1.0)

        # Canada-orange: 20 * 0.2 = 4.0
        self.assertAlmostEqual(bill.data[0, 1], 4.0)

        # Remaining elements should be 0
        self.assertAlmostEqual(bill.data[0, 2], 0.0)
        self.assertAlmostEqual(bill.data[0, 3], 0.0)

        # Brazil-apple: 0 * 0.3 = 0.0
        self.assertAlmostEqual(bill.data[1, 0], 0.0)

        # Brazil-orange: 0 * 0.4 = 0.0
        self.assertAlmostEqual(bill.data[1, 1], 0.0)

        # The multiplications only happen where both arrays have coordinates
        self.assertAlmostEqual(bill.data[1, 2], 0.0)
        self.assertAlmostEqual(bill.data[1, 3], 0.0)


if __name__ == '__main__':
    settings.data_type = 'dense'
    unittest.main()
