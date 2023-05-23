import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

df = pd.read_csv('data.csv')
df['has_symptoms'] = df['symptom_onset'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['recovered'] = df['recovered'].apply(lambda x: 0 if x == 0 or pd.isnull(x) else 1)
df['death'] = df['death'].apply(lambda x: 1 if x == 1 else 0)

#we can tell that someone really had covid if they died or they recovered from it
df['true_patient'] = df['death'] | df['recovered'] 

model = BayesianNetwork([('visiting Wuhan', 'true_patient'), ('has_symptoms', 'true_patient')])
model.fit(df, estimator=BayesianEstimator)
inference = VariableElimination(model)
result = inference.query(variables=['true_patient'], evidence={'has_symptoms': 1, 'visiting Wuhan': 1})
print(result)

