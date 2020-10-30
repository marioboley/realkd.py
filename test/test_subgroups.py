import unittest
import realkd.subgroups

from doctest import DocTestSuite


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.subgroups))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

