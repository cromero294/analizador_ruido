#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from random import *
from sklearn import tree

class ClasificadorRuido:

    def __init__(self, nepocas=100, perc=0.5):
        self.predicciones = []
        self.nepocas = nepocas
        self.perc = perc

    def setClasificadores(self, clasificadores):
        self.clasificadores = clasificadores

    def fit(self, x, y):
        clasificadores = []

        for epoca in range(self.nepocas):
            clfTree = tree.DecisionTreeClassifier()
            datos_cambiados = self.cambiarClase(x, y)
            clfTree.fit(datos_cambiados[:,:-1], datos_cambiados[:,-1])
            clasificadores.append(clfTree)

        self.setClasificadores(clasificadores)

    def score(self, x, y, clase_atrib=None):
        aciertos = 0

        for i,pred in enumerate(self.predict(x, clase_atrib)):
            if pred == y[i]:
                aciertos += 1.

        return aciertos/x.shape[0]

    def predict(self, x, clase_atrib=None):
        predicciones = []

        for pred in self.predict_proba(x, clase_atrib):
            if pred[0] > pred[1]:
                predicciones.append(0.)
            else:
                predicciones.append(1.)

        return predicciones

    def predict_proba(self, x, clase_atrib=None):
        clasificacion = []
        clasificacion_final = [[0, 0] for i in range(x.shape[0])]

        if clase_atrib == None:
            probs1 = self.predict_proba(x, 1)
            probs0 = self.predict_proba(x, 0)

            for i in range(x.shape[0]):
                clasificacion_final[i][0] += (probs0[i][0] + probs1[i][0])/2
                clasificacion_final[i][1] += (probs0[i][1] + probs1[i][1])/2

        elif clase_atrib == 1:
            datos = np.ones((x.shape[0], x.shape[1]+1))
            datos[:,:-1] = x
            x = datos
        elif clase_atrib == 0:
            datos = np.zeros((x.shape[0], x.shape[1]+1))
            datos[:,:-1] = x
            x = datos

        if clase_atrib != None:
            for clasificador in self.clasificadores:
                for i,clf in enumerate(clasificador.predict_proba(x)):
                    if clase_atrib == 1:
                        clasificacion_final[i][1] += clf[1]
                        clasificacion_final[i][0] += clf[0]
                    elif clase_atrib == 0:
                        clasificacion_final[i][1] += clf[0]
                        clasificacion_final[i][0] += clf[1]

            for i in range(x.shape[0]):
                clasificacion_final[i][0] /= len(self.clasificadores)
                clasificacion_final[i][1] /= len(self.clasificadores)

        return clasificacion_final

    def cambiarClase(self, x, y):

        datos = np.c_[x, y]

        numDatos = datos.shape[0]
        porcentaje = int(numDatos * self.perc)
        clases = datos[:,-1].copy()
        datos_nuevos = datos.copy()

        arrayAleatorio = range(0, numDatos)

        shuffle(arrayAleatorio)

        for num in arrayAleatorio[:porcentaje]:
            datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

        clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == clases[i] else 0.0) for i in range(0,numDatos)]

        return np.column_stack((datos_nuevos, np.array(clase_bien_mal_clasificado)))
