# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from Datos import Datos
import EstrategiaParticionado
import numpy as np

try:
    dataset=Datos('Datasets/wdbc.data')

    dataset.cambiarClase()

    print dataset.getDatosCambiados()
except ValueError as e:
    print e
