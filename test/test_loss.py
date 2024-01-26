import unittest
import realkd.loss
from doctest import DocTestSuite

def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.loss))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

