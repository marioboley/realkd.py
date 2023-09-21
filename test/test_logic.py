import unittest
import realkd.logic
import sys
import pathlib
from doctest import DocTestSuite


# print('Running in:', )
sys.path.append(pathlib.Path(__file__).parent.parent.resolve())
sys.path.append(pathlib.Path(__file__).parent.resolve())


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.logic))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

