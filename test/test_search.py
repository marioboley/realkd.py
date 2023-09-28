import unittest
import realkd.search.core_query_tree
import realkd.legacy.legacy

from doctest import DocTestSuite


class SearchContextUdSTestCase(unittest.TestCase):

    table = [[1, 1, 1, 1, 0],
             [1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 1, 1, 1],
             [1, 1, 0, 0, 1]]

    ctx = realkd.search.core_query_tree.SearchContext.from_tab(table)

    def test_apx_factor(self):
        labels = [1, 0, 0, 1, 1, 0]
        f = realkd.legacy.legacy.impact(labels)
        g = realkd.legacy.legacy.cov_incr_mean_bound(labels, realkd.legacy.legacy.impact_count_mean(labels))
        res = realkd.search.core_query_tree.CoreQueryTreeSearch(self.ctx, f, g, order='bestboundfirst', apx=0.5).run()
        self.assertEqual(str(res), 'c3')


def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.search.core_query_tree))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

