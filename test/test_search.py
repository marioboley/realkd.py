import unittest
import realkd.search
import realkd.legacy

from doctest import DocTestSuite


class SearchContextUdSTestCase(unittest.TestCase):

    table = [[1, 1, 1, 1, 0],
             [1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 1, 1, 1],
             [1, 1, 0, 0, 1]]

    ctx = realkd.search.Context.from_tab(table)

    def test_apx_factor(self):
        labels = [1, 0, 0, 1, 1, 0]
        f = realkd.legacy.impact(labels)
        g = realkd.legacy.cov_incr_mean_bound(labels, realkd.legacy.impact_count_mean(labels))
        res = self.ctx.search(f, g, order='bestboundfirst', apx=0.5)
        self.assertEqual(str(res), 'c3')


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.search))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

