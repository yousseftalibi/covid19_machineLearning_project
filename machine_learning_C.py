from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

df['has_symptoms'] = df['symptom_onset'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['age'].fillna(df['age'].mean(), inplace=True)
df['didRecover'] = df['recovered'].apply(lambda x: 0 if x == '0' else 1)
df['true_patient'] = (df['death'] == 1) | df['didRecover']
df['hospitalized'] = df['hosp_visit_date'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Kmeans is a distance based algorithm, so we need to only consider numeric columns.
df_cluster = df[['age', 'has_symptoms', 'hospitalized', 'true_patient']]

silhouette_scores = [] 
K = range(2, 10) # we check k from 2 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(df_cluster)
    silhouette_scores.append(silhouette_score(df_cluster, labels))

#best number of clusters. in our case we find 3.
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print(optimal_k)
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
labels = kmeans.fit_predict(df_cluster)

plt.scatter(df_cluster['age'], df_cluster['has_symptoms'], c=labels)
plt.xlabel('Age')
plt.ylabel('Has Symptoms')
plt.title('Clustering results')
plt.show()


