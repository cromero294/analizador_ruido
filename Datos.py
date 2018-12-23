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

  '''
  def transformDataset(self):
    encAtributos = preprocessing.OneHotEncoder(categorical_features=self.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(self.datos[:,:-1])
    Y = self.datos[:,-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=None, shuffle = True)

    classifier = DecisionTreeClassifier(random_state = 0)

    classifier.fit(X_train, Y_train)

    result = classifier.predict(X_test)

    new_class = [(1.0 if value == Y_test[i] else 0.0) for i,value in enumerate(result)]

    new_dataset = np.column_stack((X_test, Y_test, result, new_class))

    return new_dataset
  '''

  def cambiarClase(self, perc=0.5):
    numDatos = self.getNumDatos()
    porcentaje = int(numDatos * perc)
    clases = self.datos[:,-1]

    arrayAleatorio = range(0, numDatos)

    shuffle(arrayAleatorio)

    for num in arrayAleatorio[:porcentaje]:
        self.datos[num,-1] = 1 - self.datos[num,-1]

    #[self.datos[num,-1] <- 1 - self.datos[num,-1] for num in arrayAleatorio[:porcentaje]]
    clase_bien_mal_clasificado = [(1.0 if self.datos[i,-1] == clases[i] else 0.0) for i in range(0,numDatos)]

    self.datos_clases_cambiadas = np.column_stack((self.datos, np.array(clase_bien_mal_clasificado)))
    #self.datos_clases_cambiadas = np.hstack((self.datos,clase_bien_mal_clasificado))

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
