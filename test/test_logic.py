import unittest
import realkd.logic

from doctest import DocTestSuite


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.logic))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

