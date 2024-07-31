import unittest
from karray import settings
from tests.array_initialization import TestArrayInitialization
from tests.array_operations import TestArrayOperations
from tests.array_insert import TestArrayInsert
from tests.array_filehandling import TestArrayFileHandling
from tests.array_choice import TestArrayChoice

if __name__ == '__main__':
    settings.data_obj = 'sparse'
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(test_loader.loadTestsFromTestCase(TestArrayInitialization))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestArrayOperations))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestArrayInsert))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestArrayFileHandling))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestArrayChoice))

    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)
