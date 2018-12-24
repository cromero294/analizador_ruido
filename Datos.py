#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from random import *

class Datos(object):

  TiposDeAtributos=('Continuo','Nominal')

  def __init__(self, nombreFichero):

    self.datos_clases_cambiadas = []

    self.tipoAtributos = []
    self.nominalAtributos = []
    self.diccionarios = []

    listAux = []
    lista_de_listas = []

    try:
      fl = open (nombreFichero, "r")
      linea = fl.read().replace("\r", "").split("\n")
    except IOError:
      print ("Error: El archivo \"" + nombreFichero + "\" no se pudo abrir.")
      sys.exit(0)

    self.nDatos = int(linea[0])
    self.nombreAtributos = linea[1].split(",")

    #Configuracion tipoAtributos y nominalAtributos
    for word in linea[2].split(","):
      if word == "Continuo":
        self.tipoAtributos.append(Datos.TiposDeAtributos[0])
        self.nominalAtributos.append(False)
      elif word == "Nominal":
        self.tipoAtributos.append(Datos.TiposDeAtributos[1])
        self.nominalAtributos.append(True)
      else:
       raise ValueError('El tipo de dato \"' + word + '\" no es valido')

      self.diccionarios.append({})
      listAux.append([])

    #Creacion de listas con los distintos tipos de datos nominales y relleno de tuplas de datos
    for i in range(self.nDatos):
      datosAux = linea[i+3].split(",")
      lista_de_listas.append(datosAux)
      for j in range(len(datosAux)):
        if datosAux[j] not in listAux[j]:                                       #Si no estaba previamente
          if self.tipoAtributos[j] == Datos.TiposDeAtributos[1]:                #Si nominal
            listAux[j].append(datosAux[j])                                      #Agregamos

    #Configuracion de diccionarios discretizando los valores nominales
    for i in range(len(listAux)):
      listAux[i] = sorted(listAux[i])
      for j in range(len(listAux[i])):
        self.diccionarios[i].update({listAux[i][j]:len(self.diccionarios[i])})   #Agregamos y damos valor

    #Sustitucion de variables nominales por los valores de los diccionarios
    for i in range(len(lista_de_listas)):
      for j in range(len(self.diccionarios)):
        if self.nominalAtributos[j]:
          lista_de_listas[i][j] = float(self.diccionarios[j].get(lista_de_listas[i][j]))
        else:
          lista_de_listas[i][j] = float(lista_de_listas[i][j])

    self.datos = np.array(lista_de_listas)

    fl.close()

  def cambiarClase(self, perc=0.5):
    '''
    Funcion que hace un swap de la clase al porcentaje indicado del total de
    ejemplos de la base de datos y, una vez realizado lo dicho, genera una nueva
    clase comparando cada clase original con el nuevo conjunto de datos (con ruido aleatorio)
    que indica con un 1 si no ha cambiado y con un 0 si la clase ha cambiado.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas de la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    '''
    numDatos = self.getNumDatos()
    porcentaje = int(numDatos * perc)
    clases = self.datos[:,-1].copy()
    datos_nuevos = self.datos.copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == clases[i] else 0.0) for i in range(0,numDatos)]

    self.datos_clases_cambiadas = np.column_stack((datos_nuevos, np.array(clase_bien_mal_clasificado)))

  def cambiarClase_unos(self, perc=0.5):
    '''
    Aproximacion del metodo cambiarClase que no compara, a la hora de
    seleccionar la nueva clase (1 si bien clasificado, 0 si mal), con la clase
    original sino con 1.0.

    Los nuevos datos se guardan en la variable datos_clases_cambiadas_unos de
    la clase Datos.

    @param perc: indica el porcentaje de datos que se van a modificar.
    '''
    numDatos = self.getNumDatos()
    porcentaje = int(numDatos * perc)
    clases = self.datos[:,-1].copy()
    datos_nuevos = self.datos.copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 1.0 else 0.0) for i in range(0,numDatos)]

    self.datos_clases_cambiadas_unos = np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))

  def cambiarClase_ceros(self, perc=0.5):
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
    clases = self.datos[:,-1].copy()
    datos_nuevos = self.datos.copy()

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

    clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == 0.0 else 0.0) for i in range(0,numDatos)]

    self.datos_clases_cambiadas_ceros = np.column_stack((datos_nuevos[:,:-1], np.array(clase_bien_mal_clasificado)))

################################################################################
#                                                                              #
#                            getters y setters                                 #
#                                                                              #
################################################################################

  def getDatosCambiados(self):
    return self.datos_clases_cambiadas

  def getTipoAtributos(self):
    return self.tipoAtributos

  def getNombreAtributos(self):
    return self.nombreAtributos

  def getNominalAtributos(self):
    return self.nominalAtributos

  def getDiccionarios(self):
    return self.diccionarios

  def getDatos(self):
    return self.datos

  def getNumDatos(self):
    return self.nDatos

  def extraeDatos(self, idx):
    return self.datos[idx,:]
