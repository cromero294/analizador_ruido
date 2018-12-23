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

	transformed_dataset=dataset.transformDataset()
	transformed_dataset.changeClass(0.5)

	print transformed_dataset
except ValueError as e:
    print e
