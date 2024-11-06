# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:33:35 2024

@author: jfgonzalez
"""

import pandas as pd
from scipy.stats import wilcoxon
import numpy as np

# %%

def levanto_semillas():
    basepath='C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/vm_logs/EV/'
    files = ['WUBA_de_fabrica_base_03_30.txt', 'WUBA_de_fabrica_clust_03_37.txt',
             'Bagging_base_02_41.txt', 'Bagging_clust_01_51.txt', 
             'Otros_base_03_30.txt', 'Otros_clust_01_42.txt']
    
    columnas = ['fecha', 'rank', 'iteracion_bayesiana', 'qsemillas', 'semilla', 'corte', 'ganancia', 'metrica']
    
    lista_nombres = []
    lista_seires = []
    for file in files:
        serie = pd.read_csv(file, names=columnas, sep='\t').ganancia
        
        lista_nombres.append(file[:-10])
        lista_seires.append(serie[:-1])
        
    return pd.concat(lista_seires, axis=1, keys=lista_nombres)

df_semillas = levanto_semillas()

# %%
# Tests

print(f'·{"dataset base":22}·{"dataset clusters":22}   ·{"p valor":20}')
for i in [0, 2, 4]:
    col1, col2 = df_semillas.columns[i], df_semillas.columns[i+1]
    wil = wilcoxon(x=df_semillas[col1], y=df_semillas[col2])
    print(f' {col1:22} vs {col2:22} p.value: {wil.pvalue:,.10f}')

col1, col2 = df_semillas.columns[0], df_semillas.columns[2]
wil = wilcoxon(x=df_semillas[col1], y=df_semillas[col2])
print(f' {col1:22} vs {col2:22} p.value: {wil.pvalue:,.10f}')


# %%

basepath ='C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/vm_logs/EV/'

files_con_cortes = [
    'expw_EV-0007_ganancias_03_030.txt', 'expw_EV-0008_ganancias_03_037.txt',
    'expw_EV-0009_ganancias_03_036.txt', 'expw_EV-0010_ganancias_03_048.txt',
    'expw_EV-0007_ganancias_03_030.txt', 'expw_EV-0011_ganancias_03_035.txt']

exp_ = [3,3,2,1,3,1]

df_ = pd.read_csv( basepath+files_con_cortes[0],  sep='\t')

