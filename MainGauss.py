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

    # plt.scatter(np.array(xt1), np.array(xt2), marker = 'o', c=np.array(yt), cmap=matplotlib.colors.ListedColormap(colors))

    fig, axs = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

    num = [1, 11, 101]

    for i in range(len(axs)):
        num_arboles = num[i]

        clf = ClasificadorRuido(num_arboles)
        clf.fit(X, y)

        im1 = plotModel_arboles(np.array(xt1),np.array(xt2),np.array(yt),clf,axs[i],0,1)
        fig.colorbar(im1, ax=axs[i], ticks=range(0,20), label='class')

    # plt.show()
    plt.savefig("Imagenes/gaussiana_3_0.eps")

except ValueError as e:
    print(e)
