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

    X,y=make_moons(n_samples=100, shuffle=True, noise=0.5, random_state=None)

    datostrain = np.column_stack((X, y))

    Xt,yt=make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=None)

    datostest = np.column_stack((Xt, yt))

    cambiaClase = Clasificador()

    score_final_tree = 0

    iteracion = 1

    print("Iteracion: " + str(iteracion))

    clfTree,datos_cambiados = cambiaClase.entrenamiento(datostrain, 100, 0.5)

    score_final_tree += clfTree.score(datostest)
    clasificaciones = clfTree.predict(datostest)

    print("Arbol de decision: " + str(score_final_tree))
    print("Clasificaciones: " + str(clasificaciones))

    ####################################
    ##########     PLOT     ############
    ####################################

    print("------------PLOT------------")

    '''
    Datos en rojo, clase 1
    Datos en azul, clase 0
    '''
    plt.subplot(2, 2, 1)
    plotPuntos(datostrain, "Datos originales")

    '''
    Datos en rojo, clase 1
    Datos en azul, clase 0
    '''
    plt.subplot(2, 2, 2)
    plotClases(datos_cambiados, "Datos con ruido artificial")

    '''
    Datos en verde, dato no cambiado
    Datos en rojo, ruido artificial
    '''
    plt.subplot(2, 2, 3)
    plotPuntosRuido(datos_cambiados, "Bien/mal clasificado")

    '''
    Datos en verde, bien clasificado
    Datos en rojo, ruido
    '''
    plt.subplot(2, 2, 4)
    plotPuntosClasificados(datostest,clfTree, "Clasificacion final")

    plt.show()
    #plt.savefig(".eps")

except ValueError as e:
    print(e)
