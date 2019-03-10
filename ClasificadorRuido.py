#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from random import *
from sklearn import tree
from scipy import stats

class ClasificadorRuido:

    def __init__(self, nepocas=100, perc=0.5):
        self.nepocas = nepocas
        self.perc = perc

    def fit(self, x, y):
        self.clasificadores = []

        for epoca in range(self.nepocas):
            clfTree = tree.DecisionTreeClassifier()
            X_cambiado, y_cambiado = self.cambiarClase(x, y)
            clfTree.fit(X_cambiado, y_cambiado)
            self.clasificadores.append(clfTree)

    def score(self, x, y, clase_atrib=None):
        aciertos = 0

        pre = self.predict(x, clase_atrib)

        print pre

        for i,pred in enumerate(pre):
            if pred == y[i]:
                aciertos += 1.

        return aciertos/x.shape[0]

    def predict(self, x, clase_atrib=None):
        prediccs = []

        for pred in self.predict_proba(x, clase_atrib):
            if pred[0] > pred[1]:
                prediccs.append(0.)
            else:
                prediccs.append(1.)

        return prediccs

    def predict_proba(self, x, clase_atrib=None, save=False):
        clasificacion = []
        clasificacion_final = [[0, 0] for i in range(x.shape[0])]

        if clase_atrib == None:

            probs1 = self.predict_proba(x, 1, save)
            probs0 = self.predict_proba(x, 0, save)

            if save:
                self.pred = []
                print "----------------------------------"
                print self.pred_ceros, probs0
                print self.pred_unos, probs1
                print "----------------------------------"
                for z,y in zip(self.pred_ceros,self.pred_unos):
                    predaux = []
                    for i in range(len(z)):
                        #print z,y
                        predaux.append([(z[i][1] + y[i][0])/2, (z[i][0] + y[i][1])/2])
                        #print predaux
                    self.pred.append(predaux)
                #print self.pred
            #print probs1

            for i in range(x.shape[0]):
                clasificacion_final[i][0] += (probs0[i][0] + probs1[i][0])/2
                clasificacion_final[i][1] += (probs0[i][1] + probs1[i][1])/2

        else:

            if clase_atrib == 1:
                datos = np.ones((x.shape[0], x.shape[1]+1))
                self.pred_unos = []
            elif clase_atrib == 0:
                datos = np.zeros((x.shape[0], x.shape[1]+1))
                self.pred_ceros = []

            datos[:,:-1] = x
            x = datos

            pred = []

            for clasificador in self.clasificadores:
                aux = clasificador.predict_proba(x)
                for i,clf in enumerate(aux):
                    if clase_atrib == 1:
                        clasificacion_final[i][1] += clf[1]
                        clasificacion_final[i][0] += clf[0]
                    elif clase_atrib == 0:
                        clasificacion_final[i][1] += clf[0]
                        clasificacion_final[i][0] += clf[1]
                if save:
                    if clase_atrib == 1:
                        self.pred_unos.append(aux)
                    elif clase_atrib == 0:
                        self.pred_ceros.append(aux)

            for i in range(x.shape[0]):
                clasificacion_final[i][0] /= len(self.clasificadores)
                clasificacion_final[i][1] /= len(self.clasificadores)

        return clasificacion_final

    def score_error(self,datos,clases,nclasificadores=None):
        if nclasificadores == None:
            nclasificadores = len(self.clasificadores)

        aciertos = 0

        for i in range(datos.shape[0]):
            predic = []

            for n in range(nclasificadores):
                predic.append(self.predicciones[n][i])

            #print stats.mode(predic)[0][0]

            print stats.mode(predic)[0][0]
            if stats.mode(predic)[0][0] == clases[i]:
                aciertos+=1.0

        return aciertos/datos.shape[0]

    def predict_error(self, x, clase_atrib=None):
        aux = []

        self.predict_proba(x, clase_atrib, save=True)

        if clase_atrib == 1:
            predaux = self.pred_unos
        if clase_atrib == 0:
            predaux = self.pred_ceros
        if clase_atrib == None:
            predaux = self.pred

        for pred in predaux:
            predicciones = []
            for dato in pred:
                if clase_atrib == 0:
                    if dato[1] > dato[0]:
                        predicciones.append(0.)
                    else:
                        predicciones.append(1.)
                else:
                    if dato[0] > dato[1]:
                        predicciones.append(0.)
                    else:
                        predicciones.append(1.)
            aux.append(predicciones)

        self.predicciones = aux
        print aux
        print "----------------------------------"

    def cambiarClase(self, x, y):

        datos = np.c_[x, y]

        numDatos = datos.shape[0]
        porcentaje = int(numDatos * self.perc)

        datos_nuevos = datos.copy()

        arrayAleatorio = range(0, numDatos)

        shuffle(arrayAleatorio)

        for num in arrayAleatorio[:porcentaje]:
            datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

        clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == datos[i,-1] else 0.0) for i in range(0,numDatos)]

        return datos_nuevos, np.array(clase_bien_mal_clasificado)
