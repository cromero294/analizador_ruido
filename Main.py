# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from Datos import Datos
import EstrategiaParticionado
from sklearn import *
from sklearn.model_selection import *
from sklearn.tree import *
import numpy as np
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

try:
	dataset=Datos('example1.data')

	encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
	X = encAtributos.fit_transform(dataset.datos[:,:-1])
	Y = dataset.datos[:,-1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=None, shuffle = True)

	classifier = DecisionTreeClassifier(random_state = 0)

	classifier.fit(X_train, Y_train)

	result = classifier.predict(X_test)

	new_class = [(1.0 if value == Y_test[i] else 0.0) for i,value in enumerate(result)]

	new_dataset = np.column_stack((X_test, Y_test, result, new_class))

	print new_dataset
except ValueError as e:
    print e