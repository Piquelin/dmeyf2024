# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:02:07 2024

@author: jfgonzalez
"""

import pandas as pd
import seaborn as sns


df = pd.read_csv('C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/HT2810/gridsearch.txt', sep='\t')

sns.pairplot (df.iloc[:,:-2], hue='ganancia_mean')
