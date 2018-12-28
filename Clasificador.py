#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from random import *

class Clasificador:

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
    numDatos = self.getNumDatos()
    porcentaje = int(numDatos * perc)
    datos_nuevos = datos.getDatos().copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 0.0 else 0.0) for i in range(0,numDatos)]

    datos.datos_clases_cambiadas_ceros = np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))
