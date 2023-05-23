from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data.csv')

df['has_symptoms'] = df['symptom_onset'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['age'].fillna(df['age'].mean(), inplace=True)
df['from Wuhan'].fillna(0, inplace=True)
df['visiting Wuhan'].fillna(0, inplace=True)
predictors = ['gender', 'age', 'has_symptoms', 'visiting Wuhan', 'from Wuhan',  'country', 'location' ]

X = pd.get_dummies(df[predictors])  
y = (df['death'] == 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


param_grid = {
    'n_neighbors': [3, 6, 9, 12, 15, 18],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

clf = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print(best_parameters)
clf_best = KNeighborsClassifier(**best_parameters)
clf_best.fit(X_train, y_train)
y_pred_best = clf_best.predict(X_test)
f1_best = f1_score(y_test, y_pred_best)
print(f"\nF1 Score including location: {f1_best}")

samples = [
    {
        'gender': 'male',
        'age': 61,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei',
        
    },
    {
        'gender': 'male',
        'age': 69,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 89,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 89,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 66,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 75,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'female',
        'age': 48,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 82,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 66,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'male',
        'age': 81,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    },
    {
        'gender': 'female',
        'age': 82,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei'
    }
]

for sample in samples:
    sample_df = pd.get_dummies(pd.DataFrame([sample]))
    sample_df = sample_df.reindex(columns = X_train.columns, fill_value=0)
    outcome = clf_best.predict(sample_df)
    print('Predicted outcome:', 'died' if outcome[0] else 'discharged')
