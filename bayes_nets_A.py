import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

df = pd.read_csv('data.csv')

df['has_symptoms'] = df['symptom_onset'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['visiting Wuhan'] = df['visiting Wuhan'].apply(lambda x: 0 if (x == 0 or pd.isnull(x)) else 1)

model = BayesianNetwork([('visiting Wuhan', 'has_symptoms')])
model.fit(df, estimator=BayesianEstimator)
inference = VariableElimination(model)
result = inference.query(variables=['has_symptoms'], evidence={ 'visiting Wuhan': 1})
print(result)

