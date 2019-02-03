import Conjunto
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Autor Luis Lago y Manuel Sanchez Montanes
# Modificada por Gonzalo
def plotModel(x,y,clase,clf,title,diccionarios):
    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2

    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if isinstance(clf, Conjunto.Conjunto):
        print np.c_[xx.ravel(), yy.ravel()].shape
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    elif hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]


    z = z.reshape(xx.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #ax = plt.subplot(1, 1, 1)
    plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
    plt.contour(xx, yy, z, [0.5], linewidths=[2], colors=['k'])

    if clase is not None:
        plt.scatter(x[clase==0], y[clase==0], c='#FF0000')
        plt.scatter(x[clase==1], y[clase==1], c='#0000FF')
    else:
        plt.plot(x,y,'g', linewidth=3)

    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

def plotPuntos(datos):
    for dato in datos:
        if dato[-1] == 1.0:
            plt.plot(dato[0],dato[1],'ro')
        else:
            plt.plot(dato[0],dato[1],'bs')

    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.title('Ruido artificial')
    plt.show()

def plotPuntosClasificados(datos,clasificador):
    predicciones = clasificador.predict(datos)

    for i,dato in enumerate(datos):
        if predicciones[i] == 1.0:
            plt.plot(dato[0],dato[1],'ro')
        else:
            plt.plot(dato[0],dato[1],'bs')
            
    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.title('Clasificacion de ruido')
    plt.show()
