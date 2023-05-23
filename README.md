Youssef TALBI <br>
Introduction to artificial intelligence <br>
ISEP – 2022/2023 <br>

Epidemiological data is needed during emerging epidemics to best monitor and anticipate spread of infection.
The dataset has been made available publicly as of 20th January, 2020 containing different information about the patients: clinic, demographic and geographic. in a Github repository https://github.com/beoutbreakprepared/nCoV2019

The goal of this project is to process this dataset using artificial intelligence methods in order to help the community to better understand the spread of the COVID-19 infection. 

The project contains the following 4 parts: 
1. Analysis of the dataset :
In order to analyse the dataset, you have to extract some statistical information from the given dataset, for example: the type of data, the missing values, outliers, the correlation between  variables, etc. If there are missing values, you can replace them by the mean, median or mode of the concerning variable.

2. Bayes Nets 
In this part we will use Bayes model to compute some predictions from the data sets

3. Machine Learning 
In this part we use a machine learning method KNN in order to predict the outcome: patients outcome as either ‘died’ or ‘discharged’ from hospital. 

4. Improving the results and Theoretical formalism



COVID-19 Infection Analysis and Prediction project

1. Analysis of the dataset: 
 
A Compute the correlations between the variables. Which variables are most correlated with the target (outcome) ? Explain the results
 
 
For this question, we only consider numerical columns. We find:

 ![image](https://github.com/yousseftalibi/covid19_machineLearning_analysis-prediction/assets/58918607/5f28810b-cae0-48d7-abba-ea455bc1111c)
 



| Variable              | Correlation with Outcome |
|-----------------------|--------------------------|
| id                    | -0.169122                |
| case_in_country       | -0.026920                |
| age                   | 0.213598                 |
| If_onset_approximated | -0.013665                |
| visiting Wuhan        | -0.049711                |
| from Wuhan            | 0.370163                 |
| death                 | 1.000000                 |

 

The targets most correlated with the outcome 'death' are 'from Wuhan' and 'age'. This is natural since Covid-19 has first appeared in Wuhan and is known to be more fatal for older population. This is an old dataset, referencing the early days of Covid-19 when it was mainly contained in Wuhan.
 

B. Plot the dataset using scatter and analyse the obtained result. Use the PCA (Principal Component Analysis) to project the dataset.
 
Before plotting, we standardize so they all have a mean of 0 and standard deviation of 1. We do this with the library StandardScaler from sklearn.
 
the data Here we plot the age of people in function of their outcome 'death'. We find evenly distributed results in outcome 'survived' but almost all the 'dead' outcome are older than 40 years old. This points to the known fact that Covid-19 is rarely fatal to younger population.
 
 ![image](https://github.com/yousseftalibi/covid19_machineLearning_analysis-prediction/assets/58918607/d89e8062-8d94-4434-a07a-e576572d523a)

 
Next, we look at the date when symptoms appeared in function to the outcome. We see that in the early days the spread of Covid-19 almost all outcomes are 'death', as time went by, medical practionners learned how to deal best with the virus. This may be the reason why people survived almost always. It may also be due to better habits from people, who with the passing of time, learnt to wear masks and wash their hands. 
 
 ![image](https://github.com/yousseftalibi/covid19_machineLearning_analysis-prediction/assets/58918607/75078e8f-b008-4e16-9928-5dc5f354682a)

 
Finally we conclude this part with this PCA plot:
 
 ![image](https://github.com/yousseftalibi/covid19_machineLearning_analysis-prediction/assets/58918607/59e64642-01b4-497b-8faf-f27986d6c25d)


2.	Bayes Nets:
 
A. What is the probability for a person to have symptoms of COVID-19 (symptom_onset=date) if this person visited Wuhan (visiting Wuhan = 1) ? Consider that (symptom_onset=N/A) means that the patient is asymptomatic
 
We create a new binary column 'has_symptoms' that is 0 (false) when the variable symptom_onset is empty and 1 (true) otherwise. This corresponds to the instruction: '(symptom_onset=N/A) means that the patient is asymptomatic'
We infer this new column from our model with the evidence 'visiting_wuhan' and we find:

| has_symptoms    | phi(has_symptoms) |
|-----------------|-------------------|
| has_symptoms(0) | 0.1471            |
| has_symptoms(1) | 0.8529            |
 
Meaning, if a person visited Wuhan, there's an 85% probability they have symptoms of Covid-19.
 
 
B. What is the probability for a person to be a true patient if this person has symptoms of COVID-19 (symptom_onset=date) and this person visited Wuhan?
 
A patient of Covid-19 either dies from it or recovers from it. We create a new binary column 'true_patient' 
We create a new binary column: 'true_patient' that corresponds to variable 'death' union with the variable 'recovered'.
 
We have many empty values in variable 'recovered' and 'death'. We make the assumption that when the patient has recovered, there is less motivation to report this information, leading to empty values. Therefore, for every value different than 0, we fill it with 1.
As for variable 'death', it has a tremendous amount of empty values. We suppose that we the patient dies, there's a strong motivation to report them information, therefore we assume that all empty values in 'death' column actually correspond to cases where persons did not die.
 
We apply the model to inder this new variable with evidence: 'visiting wuhan' = 1 and 'has symptoms' = 1 and we find:
 
| true_patient    | phi(true_patient) |
|-----------------|-------------------|
| true_patient(0) | 0.0862            |
| true_patient(1) | 0.9138            |

This means that the probability for a person to be a true patient if they have symptoms of Covid-19 and they visited Wuhan is 91%. 
 

C. What is the probability for a person to death if this person visited Wuhan?
 
Using our Bayesian model, we find:
 

| death    | phi(death) |
|----------|------------|
| death(0) | 0.7353     |
| death(1) | 0.2647     |
 
Meaning, there's a 26% probability of death when a person visited Wuhan. To understand this we can come back to the variables most correlated with death:

 | Variable              | Correlation with Outcome |
|-----------------------|--------------------------|
| id                    | -0.169122                |
| case_in_country       | -0.026920                |
| age                   | 0.213598                 |
| If_onset_approximated | -0.013665                |
| visiting Wuhan        | -0.049711                |
| from Wuhan            | 0.370163                 |
| death                 | 1.000000                 |
 
We can see that, in our dataset, visitng Wuhan isn't indicative of death variable. Therefore, there's only a 26% probability of death of people who visited it.
 
 
D. Estimate the average recovery interval for a patient if this person visited Wuhan?
 
We can study the variables 'symptom_onset' and 'recovery_date' as the recovery interval with the variable 'visiting wuhan' and we find:
the number of patients who have recovered and visited Wuhan is 17
the average recovery interval for patients who visited Wuhan is 20.6875 days
 

3.	Machine Learning:

In this part we use a machine learning method in order to predict the outcome: patients outcome as either ‘died’ or ‘discharged’ from hospital. You can use the K-Nearest Neighbours (K-NN) or Bayes Classification
 
A. The obtained results should be validated using some external indexes as prediction error (Confusion matrix and Accuracy) or others as Recall, F-Meseaure,… The obtained results should be analysed in the report and provide a solution to ameliorate the results.
 
We choose KNN algorithm and we validate our results with F-Measure index.  
We use these values to predict the outcome:
predictors = ['gender', 'age', 'has_symptoms', 'visiting Wuhan', 'from Wuhan',  'country' ]
 
We train the model and we get an F1 score of 57%. Let's test this manually. 
We pick a random sample comprised of 10 patients who have 'died'. For example:

 ```python
 {
        'gender': 'male',
        'age': 61,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China'
    },
  ```
We apply the KNN model to infer the death of the sample and we find that the model predict the death accurately half of the time. We decide to improve this by adding another feature that is more distinctive. A variable like 'location' is a good choice.
 
  ```python

    {
        'gender': 'male',
        'age': 61,
        'has_symptoms': 1,
        'visiting Wuhan': 0,
        'from Wuhan': 1,
        'country': 'China',
        'location': 'Wuhan, Hubei',
    },
   ```

We feed the new predictors again to the model and we get an F1 score of 75%. We test the sample and find the model predicting one outcome more correctly. This corresponds to a 10% improvement. 
 
 
B. Use the Regression to predict the age of persons based on other variables. You have the choice on these explanatory variables? How you choose these variables? Compute the quality of the prediction using MSE error (Mean Squared Error)
 
We do not have the choice of the variables when performing the linear regression model. It only takes numerical data.  
We choose variables that might correlate with 'age', are numerical or can be transformed into numerical information. We chose:
predictors = ['has_symptoms', 'hospitalized', 'true_patient']
 
hospitalized is a new binary column, we consider that when we have empty values in column hosp_visit_date it means the person hasn't been hospitalized.
 
We find: 
 
Mean squared error: 155.57
This mean the linear regressive model is imprecise at making predictions about the age, possibly because of the limited data we have.
 
To test our data manually, we pick this sample:
   ```python

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
  ```

The model predicted: 
Predicted age is : 51.04 when person actually aged 66 years old.
Predicted age is :49.17 when person actually aged 44 years.
 


C. Apply a clustering method (K-means) on the dataset to segment the persons in different clusters. Use the Silhouette index to find out the best number of clusters. Plot the results using scatter to visually analyse the clustering structure.
 
 
Kmeans is a distance based algorithm, so we need to only consider numeric columns. We considered:
['age', 'has_symptoms', 'hospitalized', 'true_patient']
 
Using the Silhouette index we check from 2 to 10 and we find 3 to be the best number of clusters. 
We run the clustering and assign a color to each cluster and then plot the variables 'has symptoms' and 'age'.
We get:

![image](https://github.com/yousseftalibi/covid19_machineLearning_analysis-prediction/assets/58918607/9dc42134-79a0-4d90-8ca9-748d8e087191)

 
 
From this scatter plot, we can learn that we should study patients who are symptomatic in terms of their age category, considering the category [0-40] medically different than the group [40-60] and different than the group [60-100].


4. Improving the results and Theoretical formalism
 
A. The data is unbalanced. You can balance it by reducing randomly the majority class. Assume that you extract 
randomly samples that are balanced. How the prediction results will change?
 
 
There are only 183 people who have recovered from Covid-19 (have date or 1 in column recovered) as opposed to 1423 non recoveries.  This can be explained by the interest to report a death case more than a recovery case.
We can balance the data by reducing the number of asymptomatic people, this way:
 
majority_class = df['recovered'].value_counts().idxmax()
minority_class = df['recovered'].value_counts().idxmin()
df_majority = df[df['recovered'] == majority_class]
df_minority = df[df['recovered'] == minority_class]
df_majority_undersampled = df_majority.sample(df_minority.shape[0])
df_balanced = pd.concat([df_minority, df_majority_undersampled], ignore_index=True)
df = df_balanced
 
Assuming we extract balanced samples randomly, the prediction results will be more aligned to real world observation. By removing unbalance in variables like recovered , we will discover that Covid-19 has a lower fatality rate than what we originally find with unbalanced data.
 
B. How you can better manage the missing values? 
 
We can fill missing values by making conclusions from other columns. 
Let's take variable 'death' as an example. It contains 28 missing values. 
We can study the intersection of those values with variable 'symptom_onset' and 'hosp_visit_date'.
The intersection has missing values for 21 of the 28 rows.
This means that among the 28 missing values in variable 'death', those people did not have symptoms nor were they hospitalized. It is fair to assume that they did not die. Instead of dropping the missing values, it might be better to fill them with 0.
We can also fill missing values with mean value of the column. This might be appropriate for a column like 'age'.
 
C. To find the best parameters for the models, the Grid-search algorithm can be used which is available in scikit-learn library. Explain the algorithm and use it for the learning models to find the best parameters.
 
The grid search algorithm is a technique used for finding the best parameters for a learning model in order to improve the performance. The algorithm starts with a set of possible values for each parameter we want to tune and then it trains the model on each parameter values. It then tests the performance of the model and returns the parameters that make the model perform best.
 
I can use this algorithm on KNeighborsClassifier responsible for predicting patient outcome.
We start by setting the parameter for the classifier:

   ```python
param_grid = {
    'n_neighbors': [3, 6, 9, 12, 15, 18],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
   ```

The algorithm will check all number of neighbors and metric & weights. If Euclidean distance maximizes model performance for example, it will be chosen.
 

Then:
 
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid)
 
Then we find the best parameters:
 
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
 
In our case, the algorithm finds: {'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'distance'} 
 
Which we use to retrain our classifier:
 
clf_best = KNeighborsClassifier(**best_parameters)
clf_best.fit(X_train, y_train)
 
Using these parameters, we increased the rating of F-Meseaure index from 75% to 100%. Using the sample of data extracted from the dataset, we can manually verify this with the same sample of the 10 people. All the predicted outcomes are 'death' which corresponds with the dataset.
 
D. Give the algorithmically (mathematical) formalism of the method which give the best results. Explain all the parameters of the used method and their impact on the results. Some comparison with public results should be made to conclude the project.
 
Out of K-nearest neighbors, linear regression and K-means, we find that K-nearest neighbors is the method that give the best results.
This classification algorithm works based on distance. We train the algorithm on a dataset, when it encounter new unseen data, it searches for the closest K instances to it in the training set. We call those instances 'neighbors'. The algorithm then takes most common class value. Mathematically, the algorithm calculates the distance between the variable we want to predict ( outcome, in our case) and each sample in the dataset. We can use Euclidean distance, which is a straight line, its formula  is: d = √[ (x2 – x1 )² + (y2 – y1 )²] 
Manhattan distance act as a grid, calculating at a right angle. Its formula is: |x1 - x2| + |y1 - y2|
where  (x2, x1), (y2, y1)  are the coordinates of a point X and Y.
Although we are dealing with variables and columns, calculating these distances still makes sense. We can represent them in the dimensions, for instance, the representation of: 'Person visited Wuhan and person has died' can be:
visited Wuhan = 1 and  person has died = 1 making X = 1 and Y = 1.
This is the reason why we can only use numerical values in this algorithm, we must also standardize them . Categorical values must be converted somehow before. 
This algorithm uses multiple parameters, the main ones are:
 
•	The number of neighbors: We calculate the distance to the target and we select a number of instances that are closest to that distance. This will allow us to take their mode or most common class value. We call this majority voting: the class that most of the K neighbors belong to is assigned to our target.
•	Weights: This indicates how the algorithm evaluates neighbors. It can do this uniformly, meaning it gives the same weight to every neighbor or it can apply more weights to neighbors that are closer. 
•	Metric: This sets the distance method the algorithm will use.
 
In terms of impact of these parameters, changing them affect the model's accuracy. For example, if the number of neighbors is too high then the model will mistake the class by including too many neighbors that are far away. As for the weights, if the data is not uniformly distributed, using the distance will make more sense. Our data is not uniform, this is why Grid Search algorithm determined distance to be a better parameter in weights.
 

 
Comparison with another project:
 
Let's compare our method with the method from the study: 
 
"Predicting Novel CoronaVirus 2019 with Machine Learning Algorithms" by Umang Soni, Nishu Gupta and Sakshi.
 
Available in page 290 of :
 https://www.researchgate.net/profile/A-Raj-5/publication/349813217_Rethinking_the_Limits_of_Optimization_Economic_Order_Quantity_EOQ_Using_Self_Generating_Training_Model_by_Adaptive-Neuro_Fuzzy_Inference_System/links/60523dfea6fdccbfeae92d5b/Rethinking-the-Limits-of-Optimization-Economic-Order-Quantity-EOQ-Using-Self-Generating-Training-Model-by-Adaptive-Neuro-Fuzzy-Inference-System.pdf 
 
 
The study uses KNN: n_neighbors = 5 while In our case, the Grid Search finds 15 as optimal n_neighbors parameter.
 
Their results point to : 
 
'The study showed that KNN Machine Learning Algorithm could predict the fatality status of a patient most accurately with a 100% performance. KNN showed the highest precision, recall and F1 score thereby showing that the algorithm is the most reliable one to predict the health of an individual.'
 
Which corresponds to our findings. KNN in our case (after applying Grid Search best parameters ) returns F1 score of 100% and predict the sample test data with no errors.
 
With our data, we find 43 deaths only. 30 of which are male. This makes male deaths 30/42 = 71% of total deaths.
This corresponds with the study's finding: 'The number of male deaths that occurred accounted for 72.2%.'.
 
Overall, we find the same results and make the same conclusion as the study, the best performant machine learning algorithm for predicating Covid-19 death is K-nearest neighbor algorithm.

