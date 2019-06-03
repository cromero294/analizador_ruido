# -*- coding: utf-8 -*-

from __future__ import division
from ClasificadorRuido import *
import numpy as np
import random
from plotModel import *
from Datos import *
from sklearn.datasets import make_moons, make_circles, make_classification

import matplotlib.pyplot as plt
import matplotlib

try:
    x1,x2,y = createDataSet(500,"twonorm",ymargin=0.0,noise=0.2,output_boundary=False)
    xt1,xt2,yt = createDataSet(100,"twonorm",ymargin=0.0,noise=0.2,output_boundary=False)

    X = np.c_[x1, x2]
    Xt = np.c_[xt1, xt2]

    colors = ['blue', 'red']
    class_atrib = [0, 1, "both"]

    # plt.scatter(np.array(xt1), np.array(xt2), marker = 'o', c=np.array(yt), cmap=matplotlib.colors.ListedColormap(colors))

    for zz in class_atrib:
        fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

        num = [1, 11, 101, 1001]

        for i in range(len(axs)):
            num_arboles = num[i]

            clf = ClasificadorRuido(num_arboles)
            clf.fit(X, y)

            im1 = plotModel_arboles(np.array(xt1),np.array(xt2),np.array(yt),clf,axs[i],zz,True)
            fig.colorbar(im1, ax=axs[i], ticks=range(0,20), label='class')

        # plt.show()
        plt.savefig("Imagenes/gaussiana_"+ str(zz) +"_both_PRUEBA.eps")

except ValueError as e:
    print(e)
