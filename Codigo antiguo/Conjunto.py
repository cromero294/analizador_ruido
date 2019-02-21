from scipy import stats
import sys

class Conjunto:

    def __init__(self,numAtb):
        self.numAtb = numAtb # Numero de atributos que debe tener el conjunto de
                             # datos a clasificar (1 atb mas que el original porque
                             # guarda la clase cambiada como atributo)
        self.predicciones = []

    def setClasificadores(self, clasificadores):
        self.clasificadores = clasificadores

    def predict_and_save(self,datos):
        if datos.shape[1] != self.numAtb:
            print("Numero de atributos incorrecto. Numero de atributos: " + str(self.numAtb) + ". Numero utilizado: " + str(datos.shape[1]))
            sys.exit()

        for clasificador in self.clasificadores:
            predic = []

            for dato in datos:
                predic.append(clasificador.predict([dato])[0])

            self.predicciones.append(predic)

    def score_and_save(self,datos,nclasificadores=None):
        if nclasificadores == None:
            nclasificadores = len(self.clasificadores)

        aciertos = 0

        for i in range(datos.shape[0]):
            predic = []

            for n in range(nclasificadores):
                predic.append(self.predicciones[n][i])

            if stats.mode(predic)[0][0] == 1.0:
                aciertos+=1.0

        return aciertos/datos.shape[0] # devuelve el porcentaje de aciertos

    def predict(self,datos):
        '''
        Calcula las clases de los datos proporcionados con el conjunto de
        clasificadores obtenido con setClasificadores().

        @param datos: datos de test para clasificar
        @return clasificacion: array con las clases clasificadas
        '''

        if datos.shape[1] != self.numAtb:
            print("Numero de atributos incorrecto. Numero de atributos: " + str(self.numAtb) + ". Numero utilizado: " + str(datos.shape[1]))
            sys.exit()

        clasificacion = []

        for dato in datos:
            predicciones = []

            for clasificador in self.clasificadores:
                predicciones.append(clasificador.predict([dato]))

            clasificacion.append(stats.mode(predicciones)[0][0][0])

        return clasificacion

    def score(self, datos, nclasificadores=None):
        '''
        Calcula el porcentaje de acierto del conjunto de clasificadores obtenido con setClasificadores()

        @param datos: datos de test para clasificar
        @return porcentaje de aciertos
        '''

        if datos.shape[1] != self.numAtb:
            print("Numero de atributos incorrecto")
            sys.exit()

        if nclasificadores == None:
            nclasificadores = len(self.clasificadores)

        aciertos = 0

        for dato in datos:
            predicciones = []

            for clasificador in self.clasificadores[:nclasificadores]:
                predicciones.append(clasificador.predict([dato]))

            if stats.mode(predicciones)[0][0][0] == 1.0:
                aciertos+=1.0

        return aciertos/datos.shape[0] # devuelve el porcentaje de aciertos
