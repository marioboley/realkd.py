import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from realkd.datasets import titanic_data


# Survived
X, y = titanic_data(True)
print(X, y)
print(a)