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

  def entrenamiento_unos_ceros(self, datos, clf_0, clf_1, nepocas=100, perc=0.5):
    '''
    Funcion basada en el entrenamiento general pero para los conjuntos de datos
    comparados con unos y ceros en vez de con la clase original de los datos.

    @param datos: datos que utilizamos para modificar y entrenar el clasificador.
    @param clasificador: es el clasificador que vamos a entrenar. P.ej: tree.DecisionTreeClassifier()
    @param nepocas: numero de veces que se va a realizar el proceso de entrenamiento.

    @return clf_0: el clasificador ya entrenado con el conjunto de datos y comparado con ceros.
    @return clf_1: el clasificador ya entrenado con el conjunto de datos y comparado con unos.
    '''

    for epoca in range(nepocas):
        datos_ceros = self.cambiarClase_ceros(datos, perc)
        clf_0 = clf_0.fit(datos_ceros[:,:-1], datos_ceros[:,-1])

        datos_unos = self.cambiarClase_unos(datos, perc)
        clf_1 = clf_1.fit(datos_unos[:,:-1], datos_unos[:,-1])

    return clf_0, clf_1

  def entrenamiento_variasClases(self, datos, clasificador, diccionario, nepocas=100, perc=0.5):
    '''

    '''

    for epoca in range(nepocas):
        datos_cambiados = self.cambiarClase_variasClases(datos, diccionario, perc)
        clasificador = clasificador.fit(datos_cambiados[:,:-1], datos_cambiados[:,-1])

    return clasificador

################################################################################
#                                                                              #
#                              cambiar clases                                  #
#                                                                              #
################################################################################
  def cambiarClase_variasClases(self, datos, diccionario, perc=0.5):
    '''

    '''
    numDatos = datos.shape[0]
    porcentaje = int(numDatos * perc)
    clases = datos[:,-1].copy() # TODO: Puede que se pueda quitar el copy
    datos_nuevos = datos.copy() # TODO: Puede que se pueda quitar el copy

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        random_key = choice(list(diccionario))
        while random_key == datos_nuevos[num,-1]:
            random_key = choice(list(diccionario))

        datos_nuevos[num,-1] = diccionario[random_key]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == clases[i] else 0.0) for i in range(0,numDatos)]

    return np.column_stack((datos_nuevos, np.array(clase_bien_mal_clasificado)))

  def cambiarClase(self, datos, perc=0.5):
    '''
    Funcion que hace un swap de la clase al porcentaje indicado del total de
    ejemplos de la base de datos y, una vez realizado lo dicho, genera una nueva
    clase comparando cada clase original con el nuevo conjunto de datos (con ruido aleatorio)
    que indica con un 1 si no ha cambiado y con un 0 si la clase ha cambiado.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas de la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
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

  def cambiarClase_unos(self, datos, perc=0.5):
    '''
    Aproximacion del metodo cambiarClase que no compara, a la hora de
    seleccionar la nueva clase (1 si bien clasificado, 0 si mal), con la clase
    original sino con 1.0.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas_unos de
    la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    '''
    numDatos = datos.shape[0]
    porcentaje = int(numDatos * perc)
    datos_nuevos = datos.copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 1.0 else 0.0) for i in range(0,numDatos)]

    return np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))

  def cambiarClase_ceros(self, datos, perc=0.5):
    '''
    Aproximacion del metodo cambiarClase que no compara, a la hora de
    seleccionar la nueva clase (1 si bien clasificado, 0 si mal), con la clase
    original sino con 0.0.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas_ceros de
    la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    '''
    numDatos = datos.shape[0]
    porcentaje = int(numDatos * perc)
    datos_nuevos = datos.copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 0.0 else 0.0) for i in range(0,numDatos)]

    return np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))
