class Conjunto:

    def __init__(self,numAtb):
        self.numAtb = numAtb # Numero de atributos que debe tener el conjunto de
                             # datos a clasificar (1 atb mas que e original porque guarda la clase cambiada como atributo)

    def setClasificadores(self, clasificadores):
        self.clasificadores = clasificadores

    def score(self, datos, clases):
        if datos.shape[1] != self.numAtb:
            sys.log("Numero de atributos incorrecto")
        if len(clases) != datos.shape[0]:
            sys.log("Numero de clases no coincide con numero de ejemplos")
