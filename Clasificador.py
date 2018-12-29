#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from random import *

class Clasificador:

################################################################################
#                                                                              #
#                              entrenamiento                                   #
#                                                                              #
################################################################################

  def entrenamiento(self, datos, clasificador, nepocas=100):
    '''
    En la funcion entrenamiento se realiza el entrenamiento del clasificador
    solicitado (de forma general para cualquier clasificador)
    con el numero de epocas especificado (default: 100).

    @param datos: datos originales que vamos a modificar.
    @param clasificador: es el clasificador que vamos a entrenar. P.ej: tree.DecisionTreeClassifier()
    @param nepocas: numero de veces que se va a realizar el proceso de entrenamiento.

    @return clf: el clasificador ya entrenado con el conjunto especificado.
    '''

    clf = clasificador()

    for epoca in range(nepocas):
        self.cambiarClase(datos)
        clf = clf.fit(datos.getDatosCambiados()[:,:-1], datos.getDatosCambiados()[:,-1])

    return clf

  def entrenamiento_unos_ceros(self, datos, clasificador, nepocas=100):
    '''
    Funcion basada en el entrenamiento general pero para los conjuntos de datos
    comparados con unos y ceros en vez de con la clase original de los datos.

    @param datos: datos que utilizamos para modificar y entrenar el clasificador.
    @param clasificador: es el clasificador que vamos a entrenar. P.ej: tree.DecisionTreeClassifier()
    @param nepocas: numero de veces que se va a realizar el proceso de entrenamiento.

    @return clf_0: el clasificador ya entrenado con el conjunto de datos y comparado con ceros.
    @return clf_1: el clasificador ya entrenado con el conjunto de datos y comparado con unos.
    '''

    clf_0 = clasificador()
    clf_1 = clasificador()

    for epoca in range(nepocas):
        self.cambiarClase_ceros(datos)
        clf_0 = clf_0.fit(datos.getDatosCambiados_ceros()[:,:-1], datos.getDatosCambiados_ceros()[:,-1])

        self.cambiarClase_unos(datos)
        clf_1 = clf_1.fit(datos.getDatosCambiados_unos()[:,:-1], datos.getDatosCambiados_unos()[:,-1])

    return clf_0, clf_1

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
    '''
    numDatos = datos.getNumDatos()
    porcentaje = int(numDatos * perc)
    clases = datos.getDatos()[:,-1].copy()
    datos_nuevos = datos.getDatos().copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == clases[i] else 0.0) for i in range(0,numDatos)]

    datos.datos_clases_cambiadas = np.column_stack((datos_nuevos, np.array(clase_bien_mal_clasificado)))

  def cambiarClase_unos(self, datos, perc=0.5):
    '''
    Aproximacion del metodo cambiarClase que no compara, a la hora de
    seleccionar la nueva clase (1 si bien clasificado, 0 si mal), con la clase
    original sino con 1.0.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas_unos de
    la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    '''
    numDatos = datos.getNumDatos()
    porcentaje = int(numDatos * perc)
    datos_nuevos = datos.getDatos().copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 1.0 else 0.0) for i in range(0,numDatos)]

    datos.datos_clases_cambiadas_unos = np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))

  def cambiarClase_ceros(self, datos, perc=0.5):
    '''
    Aproximacion del metodo cambiarClase que no compara, a la hora de
    seleccionar la nueva clase (1 si bien clasificado, 0 si mal), con la clase
    original sino con 0.0.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas_ceros de
    la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    '''
    numDatos = datos.getNumDatos()
    porcentaje = int(numDatos * perc)
    datos_nuevos = datos.getDatos().copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 0.0 else 0.0) for i in range(0,numDatos)]

    datos.datos_clases_cambiadas_ceros = np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))
