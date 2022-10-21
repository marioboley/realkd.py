import pandas as pd
from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator, logistic_loss
# from scalene import scalene_profiler

# titanic = pd.read_csv('datasets/titanic/train.csv')
# survived = titanic.Survived
# titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
flights = pd.read_csv('archive/flight_data_2018_to_2022.csv', usecols=['Quarter', 'Month', 'Dest', 'Origin', 'Operating_Airline', 'ArrDelayMinutes', 'Cancelled', 'Distance'])
# flights = pd.read_csv('archive/flight_data_2018_to_2022.csv', usecols=['Quarter', 'Month', 'Dest', 'Origin', 'Cancelled', 'Distance'])
flights = flights.sample(n=99999)
cancelled = flights.Cancelled
flights.drop(columns=['Cancelled'], inplace=True)
re = RuleBoostingEstimator(base_learner=XGBRuleEstimator(loss=logistic_loss))

# scalene_profiler.start()
re.fit(flights, cancelled.replace(0, -1))
# scalene_profiler.stop()

# print(re.rules_)