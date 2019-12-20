# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:48:28 2019

@author: Souvik.Nath
"""

# Importing necessary libraries
import warnings
import pandas as pd
from scipy.stats import randint as sp_randInt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

# Puling the dataframe
DF = pd.read_csv('Iris.csv')
print('The list of columns are:')
print(*list(DF.columns), sep=",")

# Defining the independent and the dependent variables
X = DF[DF.columns.difference(['Id', 'Species'])]
Y = LabelEncoder().fit_transform(DF['Species'])

# Splitting the dataframe into train and test
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3,
                                                stratify=Y, random_state=42)

# Initializing the necessary estimators
NM = StandardScaler() # This sklearn estimator zero centres the variables in the dataframe
DT = DecisionTreeClassifier(random_state=42) # This sklearn estimator is the Decision Tree Algorithm

# We build a pipeline object to enforce implementation and order steps in the
# project. The objective behind defining sklearn pipelines are
# to make machine learning workflows easier to read and understand
Pipe = Pipeline(steps=[('normalizer', NM), ('classifier', DT)])


# Performing cross validation for assessing the effectiveness of the model
# and eliminating probability of overfitting
Scores = cross_validate(Pipe, XTrain, YTrain, cv=3,
                        scoring=('f1_macro', 'precision_macro'),
                        return_train_score=True)

Pipe.fit(XTrain, YTrain)
print('Precision:')
print(Scores['test_precision_macro'])
print('=============================================')
print('F1-Score:')
print(Scores['test_f1_macro'])

# Random Search
# Defining the distribution to perform hyperparameter tuning through random search
Distribution = {
    'classifier__max_depth': sp_randInt(5, 10),
    'classifier__max_features': sp_randInt(1, 4),
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__min_samples_split': sp_randInt(5, 10),
    'classifier__min_samples_leaf': sp_randInt(1, 5)
}
# Initializing and fitting random search for the pipeline defined
CVRandm = RandomizedSearchCV(Pipe, Distribution, cv=3, random_state=42)
RandmSearch = CVRandm.fit(XTrain, YTrain)

# Grid Search
# Defining the distribution to perform hyperparameter tuning through grid search
param_grid = {
              'classifier__max_depth':
                  [RandmSearch.best_params_['classifier__max_depth']-1,
                   RandmSearch.best_params_['classifier__max_depth'],
                   RandmSearch.best_params_['classifier__max_depth']+1],
              'classifier__min_samples_leaf':
                  [RandmSearch.best_params_['classifier__min_samples_leaf']-1,
                   RandmSearch.best_params_['classifier__min_samples_leaf'],
                   RandmSearch.best_params_['classifier__min_samples_leaf']+1],
              'classifier__min_samples_split':
                  [RandmSearch.best_params_['classifier__min_samples_split']-1,
                   RandmSearch.best_params_['classifier__min_samples_split'],
                   RandmSearch.best_params_['classifier__min_samples_split']+1],
}
# Initializing and fitting random search for the pipeline defined
CVGrid = GridSearchCV(Pipe, param_grid, n_jobs=-1)
GridClf = CVGrid.fit(XTrain, YTrain)

TestData = XTest.copy()
TestData['y_actual'] = YTest
TestData['pred'] = GridClf.predict(XTest)

# Generating the report for the model diagnostics
Report = pd.DataFrame(classification_report(TestData['y_actual'],
                                            TestData['pred'],
                                            output_dict=True)).transpose()
display(Report)
