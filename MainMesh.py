# -*- coding: utf-8 -*-

from __future__ import division
from ClasificadorRuido import *
import numpy as np
import random
from plotModel import *
from sklearn.datasets import make_moons, make_circles, make_classification

import matplotlib.pyplot as plt

try:
    num_arboles = 100

    ###############################
    #########    MOONS    #########
    ###############################
    # X,y=make_moons(n_samples=500, shuffle=True, noise=0.5, random_state=None)
    # Xt,yt=make_moons(n_samples=1000, shuffle=True, noise=0.5, random_state=None)

    ###############################
    ########    CIRCLES    ########
    ###############################
    # X,y=make_circles(n_samples=500, noise=0.5, factor=0.5, random_state=None)
    # Xt,yt=make_circles(n_samples=1000, noise=0.5, factor=0.5, random_state=None)

    ##############################
    ########    LINEAL    ########
    ##############################
    X,y=make_classification(n_features=2, n_samples=500, n_redundant=0, n_informative=2, random_state=None, n_clusters_per_class=1)
    Xt,yt=make_classification(n_features=2, n_samples=1000, n_redundant=0, n_informative=2, random_state=None, n_clusters_per_class=1)

    clf = ClasificadorRuido()
    clf.fit(X, y)

    print("Score: " + str(clf.score(Xt, yt)))
    print("Clasificaciones: " + str(clf.predict(Xt)))

    ####################################
    ##########     PLOT     ############
    ####################################

    print("------------PLOT------------")

    plotModel(Xt[:,0],Xt[:,1],yt,clf)

    #plt.show()
    plt.savefig("Imagenes/circles_mesh100.eps")

except ValueError as e:
    print(e)
