from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pandas as pd

df = pd.read_csv('data.csv')

df['has_symptoms'] = df['symptom_onset'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['age'].fillna(df['age'].mean(), inplace=True)
df['from Wuhan'].fillna(0, inplace=True)
df['visiting Wuhan'].fillna(0, inplace=True)

predictors = ['gender', 'age', 'has_symptoms', 'visiting Wuhan', 'from Wuhan',  'country' ]

X = pd.get_dummies(df[predictors])  
y = (df['death'] == 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

f1 = f1_score(y_test, y_pred)


print(f"F1 score: {f1}")
# this shows that our model is 57% accurate, meaning predictions are correct half of the time.

#these are samples of patients that have died. We will try predicting this outcome using the model.
samples = [
    {
        'gender': 'male',
        'age': 61,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 69,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 89,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 89,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 66,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 75,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'female',
        'age': 48,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 82,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 66,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
    {
        'gender': 'male',
        'age': 81,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    }

]

for sample in samples:
    sample_df = pd.get_dummies(pd.DataFrame([sample]))
    sample_df = sample_df.reindex(columns = X_train.columns, fill_value=0)
    outcome = clf.predict(sample_df)
    print('predicted outcome:', 'died' if outcome[0] else 'discharged')

#as expected, we found that half of the patients died and half were discharged. This is because our model is 50% accurate.
#we can improve this model by including more features. let's include location 

print("\n Now let's include location ")

predictors = ['gender', 'age', 'has_symptoms', 'visiting Wuhan', 'from Wuhan',  'country', 'location' ]

X = pd.get_dummies(df[predictors])  
y = (df['death'] == 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f"\nF1 score including location: {f1}")

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
    outcome = clf.predict(sample_df)
    print('predicted outcome:', 'died' if outcome[0] else 'discharged')

# the new model, has a 75% accuracy. It predicted 10% more sample rows than before. 