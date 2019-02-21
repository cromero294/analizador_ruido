from ClasificadorRuido import ClasificadorRuido
from Datos import Datos
import EstrategiaParticionado
from sklearn.datasets import make_moons

from sklearn import tree
import numpy as np

dataset=Datos('Datasets/wdbc.data')
estrategia = EstrategiaParticionado.ValidacionSimple(1, 80)

particiones = estrategia.creaParticiones(dataset)

datostrain = dataset.extraeDatos(particiones[0].getTrain())
datostest = dataset.extraeDatos(particiones[0].getTest())

X,y=make_moons(n_samples=100, shuffle=True, noise=0.5, random_state=None)
Xt,yt=make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=None)

# datos = np.zeros((datostest.shape[0], datostest.shape[1]))
# datos[:,:-1] = datostest[:,:-1]
# datostest = datos
# print datostest

# clf = ClasificadorRuido()
# clf.fit(datostrain[:,:-1], datostrain[:,-1])
# print(clf.predict_proba(datostest[:,:-1], 1))

clf = ClasificadorRuido()
clf.fit(X, y)
print(clf.score(Xt, yt),clf.predict(Xt),clf.predict_proba(Xt))
# print(len(clf.predict_proba(Xt, 0)))
