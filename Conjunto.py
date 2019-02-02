from scipy import stats
import sys

class Conjunto:

    def __init__(self,numAtb):
        self.numAtb = numAtb # Numero de atributos que debe tener el conjunto de
                             # datos a clasificar (1 atb mas que e original porque guarda la clase cambiada como atributo)

    def setClasificadores(self, clasificadores):
        self.clasificadores = clasificadores

    def predict(self,datos):

        clasificacion = []

        for dato in datos:
            predicciones = []

            for clasificador in self.clasificadores:
                predicciones.append(clasificador.predict([dato]))

            clasificacion.append(stats.mode(predicciones)[0][0][0])

        return clasificacion

    def clasifica(self,datos,atributosDiscretos,diccionario):
        if datos.shape[1] != self.numAtb:
            print("Numero de atributos incorrecto")
            sys.exit()

        clasificacion = []

        for dato in datos:
            predicciones = []

            for clasificador in self.clasificadores:
                predicciones.append(clasificador.predict([dato]))

            clasificacion.append(stats.mode(predicciones)[0][0][0])

        return clasificacion

    def score(self, datos):
        if datos.shape[1] != self.numAtb:
            print("Numero de atributos incorrecto")
            sys.exit()

        aciertos = 0

        for dato in datos:
            predicciones = []

            for clasificador in self.clasificadores:
                predicciones.append(clasificador.predict([dato]))

            if stats.mode(predicciones)[0][0][0] == 1.0:
                aciertos+=1.0

        return aciertos/datos.shape[0] # devuelve el porcentaje de aciertos
