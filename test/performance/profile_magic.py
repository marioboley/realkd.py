from pmlb import fetch_data

import cProfile
import pstats

from realkd.rule_boosting import RuleBoostingEstimator


NUM_RULES = 10
FILENAME = f"rule_profiling_magic_{NUM_RULES}.stats"

# Returns 2 numpy arrays
magic_X, magic_y = fetch_data(
    "magic", return_X_y=True, local_cache_dir="./datasets/.cache"
)

re = RuleBoostingEstimator(num_rules=NUM_RULES)
cProfile.run("re.fit(magic_X, magic_y)", FILENAME)
# cProfile.run("rules.fit(x_train, y_train)", FILENAME)

print(re.rules_)
p = pstats.Stats(FILENAME)
