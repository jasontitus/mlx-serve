import unittest

import bench_lookup_staircut as bench
from validate_lookup_controls import check_budget_geometry


class BudgetGeometryTests(unittest.TestCase):
    @staticmethod
    def tree(rows, legal, committed=2):
        return {'kind': 'tree', 'rows': rows, 'legal_rows': legal,
                'requested_rows': 4, 'admissible': legal >= 4,
                'committed': committed}

    def test_shallow_alternatives_can_outnumber_output_tokens(self):
        check_budget_geometry('dflash', [self.tree(3, 3)], 2)
        check_budget_geometry('dflash', [self.tree(4, 7)], 2)
        with self.assertRaises(bench.BenchError):
            check_budget_geometry('dflash', [self.tree(2, 3)], 2)

    def test_branch_width_does_not_relax_published_budget(self):
        with self.assertRaises(bench.BenchError):
            check_budget_geometry('dflash', [self.tree(4, 4, committed=3)], 2)
        with self.assertRaises(bench.BenchError):
            check_budget_geometry('dflash', [self.tree(3, 3), self.tree(3, 3)], 2)

    def test_forced_tree_admissibility_uses_branch_count(self):
        wrong = self.tree(3, 3)
        wrong['admissible'] = True
        with self.assertRaises(bench.BenchError):
            check_budget_geometry('dflash', [wrong], 2)

    def test_chain_width_remains_limited_by_output_budget(self):
        for family in ('mtp', 'pld'):
            narrow = self.tree(2, 2)
            narrow['kind'] = 'chain'
            check_budget_geometry(family, [narrow], 2)
            wide = self.tree(3, 3)
            wide['kind'] = 'chain'
            with self.assertRaises(bench.BenchError):
                check_budget_geometry(family, [wide], 2)


if __name__ == '__main__':
    unittest.main()
