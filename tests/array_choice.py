
import unittest
import numpy as np
from karray import Array, settings


class TestArrayChoice(unittest.TestCase):
    def setUp(self):
        # create an Array object for testing
        index = dict(week=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                     day=[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                          0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                     trip=[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        value = [0.1, 0.3, 0.6,
                 0.5, 0.3, 0.2,
                 0.6, 0.1, 0.3,
                 0.2, 0.7, 0.1,
                 0.4, 0.4, 0.2,
                 0.7, 0.3, 0.0,
                 0.6, 0.1, 0.3,
                 0.2, 0.7, 0.1,
                 ]
        self.prob_array = Array(data=(index, value))

    def test_choice(self):
        dim = 'trip'
        new_array = self.prob_array.choice(dim, seed=123456)
        index = {'week': np.array([0, 0, 1, 1, 2, 2, 3, 3]),
                 'day': np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                 'trip': np.array([3, 1, 1, 3, 3, 1, 1, 2])}
        value = np.array([True,  True,  True,  True,  True,  True,  True,  True])
        expected_array = Array(data=(index, value))
        mask = expected_array == new_array
        # takes about 2 sec to run with sparse lib
        self.assertTrue(mask.all())


if __name__ == '__main__':
    settings.data_type = 'dense'
    unittest.main()
