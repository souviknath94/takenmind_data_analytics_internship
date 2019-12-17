# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:48:28 2019

@author: Souvik.Nath
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

class ModifiedLabelEncoder(LabelEncoder):
    
    def fit(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

le = ModifiedLabelEncoder()
dt = DecisionTreeClassifier()

pipe = Pipeline(steps=['Labelencoder', le], [])
