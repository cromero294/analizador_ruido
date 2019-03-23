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
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt

try:
    # Xt,yt=make_moons(n_samples=20000, shuffle=True, noise=0.5, random_state=None)
    # Xt,yt=make_circles(n_samples=20000, shuffle=True, noise=0.5, random_state=None)
    Xt,yt=make_classification(n_samples=20000, shuffle=True, random_state=None)
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

            # X,y=make_moons(n_samples=500, shuffle=True, noise=0.5, random_state=None)
            # X,y=make_circles(n_samples=500, shuffle=True, noise=0.5, random_state=None)
            X,y=make_classification(n_samples=500, shuffle=True, random_state=None)

            clf.fit(X, y)
            clf.predict_proba_error(Xt, clase_atrib=elem)

            for j in range(1,num_trees+1,2):
                print "\tNumero arboles " + str(j) + "/100"
                tasas_error[j-1] += 1 - clf.score_error(Xt, yt, nclasificadores=j, clase_atrib=elem)

        error_final = list(map(lambda x: x/num_trees, tasas_error))

        print "Tasa de error: " + str(error_final)
        print k
        plt.plot(range(1,101,2),error_final[::2],linestyle=linestyle[k],color=color[k])

    #plt.show()
    plt.legend(('0', '1', '0 - 1'),loc='upper right')
    plt.title("Error - Clasificadores")
    plt.savefig("Imagenes/error_classification_tres_clases.eps")

except ValueError as e:
    print e
