# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:33:35 2024

@author: jfgonzalez
"""

import pandas as pd
from scipy.stats import wilcoxon
# import numpy as np

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


def levanto_cortes():
        basepath ='C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/vm_logs/EV/'
        
        files_con_cortes = [
            'expw_EV-0007_ganancias_03_030.txt', 'expw_EV-0008_ganancias_03_037.txt',
            'expw_EV-0009_ganancias_03_036.txt', 'expw_EV-0010_ganancias_03_048.txt',
            'expw_EV-0007_ganancias_03_030.txt', 'expw_EV-0011_ganancias_03_035.txt',
            'EV-0013_pollo_parrillero_01_030.txt']
        
        exp_ = [3,3,2,1,3,1, 1]
        
        # df_ = pd.read_csv( basepath+files_con_cortes[0], sep='\t')
        
        files = ['WUBA_de_fabrica_base_03_30.txt', 'WUBA_de_fabrica_clust_03_37.txt',
                 'Bagging_base_02_41.txt', 'Bagging_clust_01_51.txt', 
                 'Otros_base_03_30.txt', 'Otros_clust_01_42.txt', 'Pollo_parrillero_01_030.txt']
            
        lista_nombres = []
        lista_series = []
        lista_envios = []
        
        for i in range(7):
            mod = exp_[i]
            lista_columnas = ['envios', f'gan_sum_{mod}']
            lista_nombres.append(files[i][:-10])
            for j in range(20):
                lista_columnas.append(f'm_{mod}_{j+1}')
            
            df_ = pd.read_csv(basepath + files_con_cortes[i], sep='\t')
            df_ = df_[lista_columnas]
            serie = df_.loc[df_[f'gan_sum_{mod}'].argmax()].reset_index(drop=True)
            envio = serie[:2]
            lista_envios.append(envio)
            lista_series.append(serie[2:])
        
        valores_corte = pd.concat(lista_series, axis=1, keys=lista_nombres)
        corte = pd.concat(lista_envios, axis=1, keys=lista_nombres)            
        return valores_corte, corte
    
df_semillas, cortes = levanto_cortes()

# df_semillas = levanto_semillas()

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


col1, col2 = df_semillas.columns[0], df_semillas.columns[6]
wil = wilcoxon(x=df_semillas[col1], y=df_semillas[col2])
print(f' {col1:22} vs {col2:22} p.value: {wil.pvalue:,.10f}')



# %%


basepath ='C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/vm_logs/'

file='SC-0020_pollo_parrillero_future_prediccion.txt'

def calcular_cortes_y_promedios(basepath, file):
    df_ = pd.read_csv(basepath+file, sep='\t')
    
    df_['prom_1'] = df_.iloc[:,3:23].T.mean()
    df_['prom_2'] = df_.iloc[:,23:43].T.mean()
    df_['prom_3'] = df_.iloc[:,43:63].T.mean()
    df_['prom_T'] = df_.iloc[:,3:63].T.mean()
    
    
    lista_series =[]
    for col in df_.columns[3:]:
        
        df_prom = df_[['numero_de_cliente', 'clase_ternaria', col]]
        
        df_prom.insert(0, 'valor', df_prom['clase_ternaria'].map(lambda x: 273000 if x == "BAJA+2" else -7000))
        df_prom = df_prom.sort_values(col, ascending=False)
        df_prom['ganancia'] = df_prom['valor'].cumsum()
        df_prom = df_prom.reset_index()
        df_prom['ganancia'].argmax()
        print( f'columna: {col}', 'corte:', df_prom['ganancia'].argmax(), 'ganancia:',df_prom.loc[df_prom['ganancia'].argmax()]['ganancia'])
        lista_series.append(df_prom['ganancia'])
        del df_prom
    ganancias = pd.concat(lista_series, axis=1, keys=df_.columns[3:])
    return ganancias

ganancias = calcular_cortes_y_promedios(basepath, file)

del i


df_graf = ganancias.iloc[8000:16000]

df_graf.columns

modelo = 1

import matplotlib.pyplot as plt

ax = plt.plot(df_graf.iloc[:, 0:20], c='grey', alpha=0.5)
plt.plot(df_graf.iloc[:,61], c='red', alpha=1,)


ax = plt.plot(df_graf.iloc[:, 20:40], c='grey', alpha=0.5)
plt.plot(df_graf.iloc[:,62], c='red', alpha=1,)


ax = plt.plot(df_graf.iloc[:, 40:60], c='grey', alpha=0.5)
plt.plot(df_graf.iloc[:,63], c='red', alpha=1,)

