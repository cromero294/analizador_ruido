# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from Datos import Datos
from Clasificador import Clasificador
from sklearn import tree
import EstrategiaParticionado
import numpy as np

try:
    '''
    En este Main se prueba la clasificacion de ruido con respecto a unos y ceros.
    Obtener 0 en el clasificador 1 significa que en el clasificador 0 necesitamos tener un 1 y viceversa.

    El score se mide suponiendo que el dataset no tiene nada de ruido.
    '''

    dataset=Datos('Datasets/wdbc.data')
    #print dataset.getDiccionarios()
    num_particiones = 20

    estrategia = EstrategiaParticionado.ValidacionCruzada(num_particiones)
    cambiaClase = Clasificador()

    particiones = estrategia.creaParticiones(dataset)

    score_final_ceros = 0
    score_final_unos = 0

    for particion in particiones:
        datostrain = dataset.extraeDatos(particion.getTrain())
        datostest = dataset.extraeDatos(particion.getTest())

        clf0, clf1 = cambiaClase.entrenamiento_unos_ceros(datostrain, tree.DecisionTreeClassifier, 100)
        score_final_ceros += clf0.score(datostest[:,:-1], np.ones(datostest.shape[0]))
        score_final_unos += clf1.score(datostest[:,:-1], np.ones(datostest.shape[0]))

    print score_final_ceros/num_particiones
    print score_final_unos/num_particiones

    #print clf.predict([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189,1.0], [13.05,19.31,82.61,527.2,0.0806,0.03789,0.000692,0.004167,0.1819,0.05501,0.404,1.214,2.595,32.96,0.007491,0.008593,0.000692,0.004167,0.0219,0.00299,14.23,22.25,90.24,624.1,0.1021,0.06191,0.001845,0.01111,0.2439,0.06289,0.0]])
except ValueError as e:
    print e
