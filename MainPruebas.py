from ClasificadorRuido import ClasificadorRuido
from Datos import Datos
import EstrategiaParticionado
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

from sklearn import tree
import numpy as np

X,y=make_moons(n_samples=500, shuffle=True, noise=0.5, random_state=None)
Xt,yt=make_moons(n_samples=2, shuffle=True, noise=0.5, random_state=None)

dataset=Datos('Datasets/example1.data')
estrategia = EstrategiaParticionado.ValidacionSimple(1, 95)
particiones = estrategia.creaParticiones(dataset)
datostrain = dataset.extraeDatos(particiones[0].getTrain())
datostest = dataset.extraeDatos(particiones[0].getTest())

clf = ClasificadorRuido(2)
clfRandom = RandomForestClassifier(n_estimators=100)

clfRandom.fit(X,y)

clf.fit(X, y)
clf.predict_error(Xt)

print yt

print "Ceros: " + str(clf.score(Xt, yt, 0))
print "Unos: " + str(clf.score(Xt, yt, 1))
print "Score: " + str(clf.score(Xt, yt))
print "Score error: " + str(clf.score_error(Xt, yt))
print "Random forest: " + str(clfRandom.score(Xt, yt))
