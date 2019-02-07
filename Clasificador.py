#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from Conjunto import *
from random import *
from sklearn import tree

class Clasificador:

################################################################################
#                                                                              #
#                              entrenamiento                                   #
#                                                                              #
################################################################################

  def entrenamiento(self, datos, nepocas=100, perc=0.5):
    '''
    Entrena un conjunto (nepocas) de clasificadores

    @param datos: datos de entrenamiento
    @param nepocas: numero de clasificadores que vamos a entrenar
    @param perc: porcentaje de ruido artificial que se incluye
    @return datos_cambiados: ultimos datos modificados para mostrarlos
    @return conjuntoClasificadores: el conjunto de clasificadores creado
    '''

    conjuntoClasificadores = Conjunto(datos.shape[1])
    clasificadores = []

    for epoca in range(nepocas):
        clfTree = tree.DecisionTreeClassifier()
        datos_cambiados = self.cambiarClase(datos, perc)
        clfTree.fit(datos_cambiados[:,:-1], datos_cambiados[:,-1])
        clasificadores.append(clfTree)
        #print id(clfTree)

    conjuntoClasificadores.setClasificadores(clasificadores)

    return conjuntoClasificadores, datos_cambiados

################################################################################
#                                                                              #
#                              cambiar clases                                  #
#                                                                              #
################################################################################
  def cambiarClase(self, datos, perc=0.5):
    '''
    Funcion que hace un swap de la clase al porcentaje indicado del total de
    ejemplos de la base de datos y, una vez realizado lo dicho, genera una nueva
    clase comparando cada clase original con el nuevo conjunto de datos (con ruido aleatorio)
    que indica con un 1 si no ha cambiado y con un 0 si la clase ha cambiado.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas de la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    @return conjunto de datos modificados
    '''
    numDatos = datos.shape[0]
    porcentaje = int(numDatos * perc)
    clases = datos[:,-1].copy() # TODO: Puede que se pueda quitar el copy
    datos_nuevos = datos.copy() # TODO: Puede que se pueda quitar el copy

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == clases[i] else 0.0) for i in range(0,numDatos)]

    return np.column_stack((datos_nuevos, np.array(clase_bien_mal_clasificado)))
