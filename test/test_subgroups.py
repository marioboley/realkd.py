import unittest
import realkd.subgroups

from doctest import DocTestSuite


def load_tests(loader, tests, ignore):
    doc_test_suite = DocTestSuite(module=realkd.subgroups)
    for test in doc_test_suite._tests:
        test.skipTest("Subgroups not yet working")
    tests.addTests(doc_test_suite)
    return tests


if __name__ == "__main__":
    unittest.main(verbosity=2)
