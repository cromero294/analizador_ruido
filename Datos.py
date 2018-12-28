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

################################################################################
#                                                                              #
#                            getters y setters                                 #
#                                                                              #
################################################################################

  def getDatosCambiados(self):
    return self.datos_clases_cambiadas

  def getDatosCambiados_unos(self):
    return self.datos_clases_cambiadas_unos

  def getDatosCambiados_ceros(self):
    return self.datos_clases_cambiadas_ceros

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
