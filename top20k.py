# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:14:41 2024

@author: Piquelin

Dividir por Scoring


"""

import pandas as pd


# %%



def armo_entregas_desde_probs(df_, modelos=3, semillas=20, entregas=False):
    lista_prob_prom = []

    for i in range(modelos):
        df_entrega =  df_[['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
        # df_entrega['numero_de_cliente'] = df_['numero_de_cliente']
        df_entrega['prom'] = df_.iloc[:,(i*semillas + 3):(i*semillas + 3 + semillas)].T.mean()
        df_entrega = df_entrega.sort_values('prom', ascending=False).reset_index()
        df_entrega = df_entrega[['numero_de_cliente',  'foto_mes', 'clase_ternaria', 'prom']]
        lista_prob_prom.append(df_entrega)

    if entregas:
        total_clientes =len(lista_prob_prom[0])
        for i in range(modelos):
            for corte in range(8000, 16001, 500):
                array = np.zeros((total_clientes, 1))
                array[:corte] = 1
                lista_prob_prom[i][f'pred_{corte}'] = array.astype(int)

    return lista_prob_prom




# %%
PATH = 'E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas/'
SC_FILE = 'exp/vm_logs/SC-0001_primero_base_future_prediccion.txt'
SC_FILE = 'exp/vm_logs/SC/expw_SC-0015_tb_future_prediccion.txt'
PQ_FILE = 'datasets/competencia_02.parquet'

# agarro cualquier archivo de Scoring con una buena predicción
df_score = pd.read_csv(PATH+SC_FILE, sep='\t')

# seleccione ol dataset base
df_comp = pd.read_parquet(PATH+PQ_FILE)

# %%

# # promedio las semillas y ordeno para el corte por probabilidad
# df_score['prom_1'] = df_score.iloc[:, 3:].T.mean()
# df_score = df_score.sort_values('prom_1', ascending=False)

lista_prom_modelos = armo_entregas_desde_probs(df_score, modelos=3, semillas=20)

# %%

# armo conjunto de clientes que me interesa top 20K de mi prediccion
estim = set(lista_prom_modelos[2].iloc[:20000]['numero_de_cliente'])

bajas = df_comp[df_comp['clase_ternaria'].isin({'BAJA+1'})]
bajas_set = set(bajas['numero_de_cliente'])

a_guardar = estim.union(bajas_set)
# separo esos del dataset
df_20k = df_comp[df_comp['numero_de_cliente'].isin(a_guardar)]



# %% Guardo

df_20k.to_csv('datasets/competencia_02_top20k_06.csv.gz', compression='gzip')
