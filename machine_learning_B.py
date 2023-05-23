from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv('data.csv')

df['has_symptoms'] = df['symptom_onset'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['age'].fillna(df['age'].mean(), inplace=True)
df['didRecover'] = df['recovered'].apply(lambda x: 0 if x == '0' else 1)
df['true_patient'] = (df['death'] == 1) | df['didRecover']
df['hospitalized'] = df['hosp_visit_date'].apply(lambda x: 0 if pd.isnull(x) else 1)

predictors = ['has_symptoms', 'hospitalized', 'true_patient']
target = 'age'

X = df[predictors]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'mean squared error: {mse}')

samples = [
    #is 66 years old in real dataset
    {
        'has_symptoms': 1,
        'hospitalized': 1,
        'true_patient': 0
    },
    #is 44 years old in real dataset
    {
        'has_symptoms': 1,
        'hospitalized': 0,
        'true_patient': 0
    }
]

for sample in samples:
    sample_df = pd.get_dummies(pd.DataFrame([sample]))
    sample_df = sample_df.reindex(columns = X_train.columns, fill_value=0)
    outcome = reg.predict(sample_df)
    print(outcome)
    print('predicted age is :', outcome[0] )
