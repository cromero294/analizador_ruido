# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from Datos import Datos
from Clasificador import Clasificador
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import EstrategiaParticionado
import numpy as np

try:
    '''
    En este Main se prueba la clasificacion de ruido con respecto a la clase original.
    Que obtengamos 0 a la hora de predecir significa que es ruido.

    El score se mide suponiendo que el dataset no tiene nada de ruido.
    '''

    dataset=Datos('Datasets/car.data')

    num_particiones = 20

    estrategia = EstrategiaParticionado.ValidacionCruzada(num_particiones)
    cambiaClase = Clasificador()

    particiones = estrategia.creaParticiones(dataset)

    #clfTree = tree.DecisionTreeClassifier()
    clfRandom = RandomForestClassifier(n_estimators=1001, max_depth=2, random_state=0)
    #clfKNN = KNeighborsClassifier(n_neighbors=101, weights = 'uniform')

    score_final_tree = 0
    score_final_random = 0
    score_final_KNN = 0

    for particion in particiones:
        datostrain = dataset.extraeDatos(particion.getTrain())
        datostest = dataset.extraeDatos(particion.getTest())

        #clfTree = cambiaClase.entrenamiento_variasClases(datostrain, clfTree, dataset.getDiccionarios()[-1], 100, 0.5)
        clfRandom = cambiaClase.entrenamiento_variasClases(datostrain, clfRandom, dataset.getDiccionarios()[-1], 100, 0.5)
        #clfKNN = cambiaClase.entrenamiento_variasClases(datostrain, clfKNN, dataset.getDiccionarios()[-1], 100, 0.5)

        #score_final_tree += clfTree.score(datostest, np.ones(datostest.shape[0]))
        score_final_random += clfRandom.score(datostest, np.ones(datostest.shape[0]))
        #score_final_KNN += clfKNN.score(datostest, np.ones(datostest.shape[0]))

    #print "Arbol de decision: " + str(score_final_tree/num_particiones)
    print "Random forest: " + str(score_final_random/num_particiones)
    #print "kNN-101: " + str(score_final_KNN/num_particiones)

    #print clf.predict([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189,1.0],
    #                   [13.05,19.31,82.61,527.2,0.0806,0.03789,0.000692,0.004167,0.1819,0.05501,0.404,1.214,2.595,32.96,0.007491,0.008593,0.000692,0.004167,0.0219,0.00299,14.23,22.25,90.24,624.1,0.1021,0.06191,0.001845,0.01111,0.2439,0.06289,0.0]])
except ValueError as e:
    print e
