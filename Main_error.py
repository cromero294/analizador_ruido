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
import random
from plotModel import *
from sklearn.datasets import make_moons

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

try:
    '''
    En este Main se prueba la clasificacion de ruido con respecto a la clase original.
    Que obtengamos 0 a la hora de predecir significa que es ruido.

    El score se mide suponiendo que el dataset no tiene nada de ruido.
    '''

    ########### DATOS ###########
    X,y=make_moons(n_samples=500, shuffle=True, noise=0.5, random_state=None)
    datostrain = np.column_stack((X, y))

    Xt,yt=make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=None)
    datostest = np.column_stack((Xt, yt))

    ########### CLASIFICADOR ###########
    cambiaClase = Clasificador()
    num_clasificadores = 501

    score_final_tree = []

    clfTree,_ = cambiaClase.entrenamiento(datostrain, num_clasificadores, 0.5)

    for i in range(1,num_clasificadores+1):
        if i%2 != 0:
            print "Iteracion " + str(i) +"/501"
            scores = []
            for x in range(100):
                print "\tSubiteracion " + str(x+1) + "/100"
                scores.append(1 - clfTree.score(datostest,i))
            score_final_tree.append(np.array(scores).mean())

    print("Arbol de decision: " + str(score_final_tree))

    ########### GRAFICAS ###########
    plt.plot(range(1,502,2),score_final_tree)
    #plt.show()
    plt.title("Error - Clasificadores (Impares)")
    plt.savefig("Imagenes/B_error_moons_impares_menos_datos.eps")

except ValueError as e:
    print(e)
