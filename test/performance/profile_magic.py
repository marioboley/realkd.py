from pmlb import fetch_data

import cProfile
from timeit import timeit
import pstats

from realkd.rule_boosting import RuleBoostingEstimator

class Benchmark:
    def fit(self):
        self.model.fit(self.X, self.y)
        
    def run_profile(self):
        cProfile.run("self.fit()", self.file_name)
        
        
    def run_benchmark(self):
        print(timeit("self.fit()", globals={"self": self}))
        

class BenchmarkMagic(Benchmark):
    def __init__(self, num_rules):
        # Returns 2 numpy arrays
        magic_X, magic_y = fetch_data(
            "magic", return_X_y=True, local_cache_dir="./datasets/.cache"
        )

        re = RuleBoostingEstimator(num_rules=num_rules)
        
        self.model = re
        self.X = magic_X
        self.y = magic_y
        self.file_name = f"rule_profiling_magic_{num_rules}.stats"

hi = BenchmarkMagic(5)
hi.run_benchmark()