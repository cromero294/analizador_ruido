# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from Datos import Datos
from ClasificadorRuido import ClasificadorRuido
from sklearn import tree
import EstrategiaParticionado
import numpy as np
import random
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

try:
    Xt,yt=make_moons(n_samples=20000, shuffle=True, noise=0.5, random_state=None)
    # datostest = np.column_stack((Xt, yt))

    clf = ClasificadorRuido()
    clase_atrib = [0, 1, None]
    linestyle = ['-.', ':', '-']
    color = ['red', 'olive', 'skyblue']

    for k,elem in enumerate(clase_atrib):
        print "-------------------ATRIB-------------------"

        iteracion = 1
        num_trees = 100
        tasas_error = [0. for x in range(num_trees)]

        for i in range(100):
            print "Iteracion " + str(i+1) + "/100"

            X,y=make_moons(n_samples=500, shuffle=True, noise=0.5, random_state=None)

            clf.fit(X, y)
            clf.predict_error(Xt, clase_atrib=elem)

            for j in range(1,num_trees+1,2):
                print "\tNumero arboles " + str(j) + "/100"
                tasas_error[j-1] += 1 - clf.score_error(Xt, yt, j)

        error_final = list(map(lambda x: x/num_trees, tasas_error))

        print "Tasa de error: " + str(error_final)
        print k
        plt.plot(range(1,101,2),error_final[::2],linestyle=linestyle[k],color=color[k])

    #plt.show()
    plt.legend(('0', '1', '0 - 1'),loc='upper right')
    plt.title("Error - Clasificadores")
    plt.savefig("Imagenes/error_moons_tres_clases.eps")

    #print clf.predict([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189,1.0],
    #                   [13.05,19.31,82.61,527.2,0.0806,0.03789,0.000692,0.004167,0.1819,0.05501,0.404,1.214,2.595,32.96,0.007491,0.008593,0.000692,0.004167,0.0219,0.00299,14.23,22.25,90.24,624.1,0.1021,0.06191,0.001845,0.01111,0.2439,0.06289,0.0]])
except ValueError as e:
    print e
