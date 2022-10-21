import pandas as pd
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator, logistic_loss

if(False):
    dataset = pd.read_csv('datasets/titanic/train.csv')
    y = dataset.Survived
    dataset.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
else:
    dataset = pd.read_csv('archive/flight_data_2018_to_2022.csv', usecols=['Quarter', 'Month', 'Dest', 'Origin', 'Operating_Airline', 'ArrDelayMinutes', 'Cancelled', 'Distance'])
    dataset = pd.read_csv('archive/flight_data_2018_to_2022.csv', usecols=['Quarter', 'Month', 'Dest', 'Origin', 'Cancelled', 'Distance'])
    dataset = dataset.sample(n=99999)
    y = dataset.Cancelled
    dataset.drop(columns=['Cancelled'], inplace=True)


re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))
re.fit(dataset, y.replace(0, -1))
print(re.rules_)
