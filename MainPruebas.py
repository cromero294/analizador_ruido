# from ClasificadorRuido import ClasificadorRuido
# from Datos import Datos
# import EstrategiaParticionado
# from sklearn.datasets import make_moons
#
# from sklearn import tree
# import numpy as np
#
# X,y=make_moons(n_samples=100, shuffle=True, noise=0.5, random_state=None)
# Xt,yt=make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=None)
#
# clf = ClasificadorRuido()
# clf.fit(X, y)
#
# print clf.score(Xt, yt, 0)
# clf.predict(Xt, 0)
# clf.predict_proba(Xt, None, True)
# clf.predict_error(Xt)
# print clf.score_error(Xt, yt)

import matplotlib.pyplot as plt

plt.plot(range(1,10),range(1,10),color='skyblue')
plt.plot(range(1,10),range(3,12),color='olive',linestyle=':')
plt.plot(range(1,10),range(5,14),color='red',linestyle='-.')


plt.show()
