import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')

df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['symptom_onset'] = pd.to_datetime(df['symptom_onset'], format='%m/%d/%Y', errors='coerce')
df['age'].fillna(df['age'].mean(), inplace=True)
df['from Wuhan'].fillna(0, inplace=True)
df['visiting Wuhan'].fillna(0, inplace=True)

numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation = df[numeric_columns].corr()
print("correlations: ")
print(correlation)

print("variables most correlated with the target (outcome): ")
print(correlation['death'])

plt.figure(figsize=(10, 8))
plt.scatter(df['death'], df['age'])
plt.xlabel('death')
plt.ylabel('age')
plt.title('death vs age')
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(df['symptom_onset'], df['death'])
plt.xlabel('syptoms date')
plt.ylabel('death')
plt.title('date of symptoms vs death')
plt.show()


scaler = StandardScaler()
df_normalized = scaler.fit_transform(df[numeric_columns].dropna())

pca = PCA(n_components=2)  
pcaResult = pca.fit_transform(df_normalized)

plt.figure(figsize=(10, 8))
plt.scatter(pcaResult[:,0], pcaResult[:,1])
plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.show()
