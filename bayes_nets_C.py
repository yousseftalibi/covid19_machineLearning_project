import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

df = pd.read_csv('data.csv')

model = BayesianNetwork([('visiting Wuhan', 'death')])
model.fit(df, estimator=BayesianEstimator)
inference = VariableElimination(model)
result = inference.query(variables=['death'], evidence={'visiting Wuhan': 1})
print(result)
