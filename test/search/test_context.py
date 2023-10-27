import unittest
import realkd.search.context
from doctest import DocTestSuite


        # >>> titanic_ctx = SearchContext.from_array(, max_col_attr=defaultdict(lambda: None, Age=6, Fare=6),
        # ...                               sort_attributes=False)
        # >>> titanic_ctx.attributes # doctest: +NORMALIZE_WHITESPACE
        # [Survived<=0, Survived>=1, Pclass<=1, Pclass<=2, Pclass>=2, Pclass>=3, Sex==male, Sex==female, Age<=23.0,
        # Age>=23.0, Age<=34.0, Age>=34.0, Age<=80.0, Age>=80.0, SibSp<=0, SibSp<=1, SibSp>=1, SibSp<=2, SibSp>=2,
        # SibSp<=3, SibSp>=3, SibSp<=4, SibSp>=4, SibSp<=5, SibSp>=5, SibSp>=8, Parch<=0, Parch<=1, Parch>=1, Parch<=2,
        # Parch>=2, Parch<=3, Parch>=3, Parch<=4, Parch>=4, Parch<=5, Parch>=5, Parch>=6, Fare<=8.6625, Fare>=8.6625,
        # Fare<=26.0, Fare>=26.0, Fare<=512.3292, Fare>=512.3292, Embarked==S, Embarked==C, Embarked==Q, Embarked==nan]
def load_tests(loader, tests, ignore):
    tests.addTests(DocTestSuite(module=realkd.search.context))
    return tests


if __name__ == '__main__':
    unittest.main(verbosity=2)

