import Conjunto
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

# Autor Luis Lago y Manuel Sanchez Montanes
# Modificada por Gonzalo
def plotModel(x,y,clase,clf,title,diccionarios):
    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2

    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if isinstance(clf, Conjunto.Conjunto):
        z1 = clf.predict(np.c_[xx.ravel(), yy.ravel(), np.ones(np.c_[xx.ravel(), yy.ravel()].shape[0])])
        z2 = clf.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros(np.c_[xx.ravel(), yy.ravel()].shape[0])])
    elif hasattr(clf, "decision_function"):
        z1 = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), np.ones(np.c_[xx.ravel(), yy.ravel()].shape[0])])
        z2 = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros(np.c_[xx.ravel(), yy.ravel()].shape[0])])
    else:
        z1 = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), np.ones(np.c_[xx.ravel(), yy.ravel()].shape[0])])[:, 1]
        z2 = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), np.ones(np.c_[xx.ravel(), yy.ravel()].shape[0])])[:, 1]

    z_list = [z1, z2, [], z1, z2, list(map(lambda x, y: x + y, z1, z2))] # Anyado z1 y z2 al final para poder pintarlas sin puntos en el mismo bucle

    for i in range(len(z_list)):
        if i == 2:
            continue

        plt.subplot(2, 3, i+1)

        z = np.array(z_list[i])
        z = z.reshape(xx.shape)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        #ax = plt.subplot(1, 1, 1)
        plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
        #plt.contour(xx, yy, z, [0.1], linewidths=[0.5], colors=['k'])

        if i == 0 or i == 1:
            if clase is not None:
                plt.scatter(x[clase==0.], y[clase==0.], marker = 'o', c='green')
                plt.scatter(x[clase==1.], y[clase==1.], marker = '_', c='yellow')
            else:
                plt.plot(x,y,'g', linewidth=3)

        plt.gca().set_xlim(xx.min(), xx.max())
        plt.gca().set_ylim(yy.min(), yy.max())
        plt.grid(True)

    plt.tight_layout()

def plotClases(datos, titulo):
    for dato in datos:
        if dato[-2] == 1.0:
            plt.plot(dato[0],dato[1],'ro')
        else:
            plt.plot(dato[0],dato[1],'bs')

    plt.title(titulo)

def plotPuntos(datos, titulo):
    for dato in datos:
        if dato[-1] == 1.0:
            plt.plot(dato[0],dato[1],'ro')
        else:
            plt.plot(dato[0],dato[1],'bs')

    plt.title(titulo)

def plotPuntosRuido(datos, titulo):
    for dato in datos:
        if dato[-1] == 1.0:
            plt.plot(dato[0],dato[1],'g^')
        else:
            plt.plot(dato[0],dato[1],'ro')

    plt.title(titulo)

def plotPuntosClasificados(datos,clasificador,titulo):
    predicciones = clasificador.predict(datos)

    for i,dato in enumerate(datos):
        if predicciones[i] == 1.0:
            plt.plot(dato[0],dato[1],'g^')
        else:
            plt.plot(dato[0],dato[1],'ro')

    plt.title(titulo)
