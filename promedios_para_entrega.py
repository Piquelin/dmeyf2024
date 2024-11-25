# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:17:22 2024

@author: Piquelin
"""

import os
import numpy as np
import pandas as pd

# %% funciones


def armo_entregas_desde_probs(df_, modelos=3, semillas=20):
    lista_prob_prom = []

    for i in range(modelos):
        df_entrega = df_[['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
        # df_entrega['numero_de_cliente'] = df_['numero_de_cliente']
        df_entrega['prom'] = df_.iloc[:, (i*semillas + 3):(i*semillas + 3 + semillas)].T.mean()
        df_entrega = df_entrega.sort_values('prom', ascending=False).reset_index()
        df_entrega = df_entrega[['numero_de_cliente',  'foto_mes', 'clase_ternaria', 'prom']]
        lista_prob_prom.append(df_entrega)

    total_clientes = len(lista_prob_prom[0])
    for i in range(modelos):
        for corte in range(8000, 16001, 500):
            array = np.zeros((total_clientes, 1))
            array[:corte] = 1
            lista_prob_prom[i][f'pred_{corte}'] = array.astype(int)

    return lista_prob_prom


def guardo_en_archivos(dfs, experimento):
    # Crear directorio de entregas
    dir_entregas = f'entregas_{experimento}'
    os.makedirs(dir_entregas, exist_ok=True)

    # Guardar predicciones de cada modelo en archivos CSV
    for modelo, df in enumerate(dfs):
        for col in df.columns:
            if col.startswith('pred_'):
                corte = col.split('_')[1]  # Extraer el número del corte
                df_pred = df[['numero_de_cliente', col]].copy()
                df_pred.columns = ['numero_de_cliente', 'Predicted']

                # Nombre de archivo y ruta de guardado
                archivo_nombre = f"{experimento}_{modelo}_{corte}.csv"
                archivo_ruta = os.path.join(dir_entregas, archivo_nombre)

                # Guardar en CSV
                df_pred.to_csv(archivo_ruta, index=False)
    return None

# %%


basepath = '../exp/vm_logs/'
file = 'SC/expw_SC-0034_tb_future_prediccion.txt'
df_ = pd.read_csv(basepath + file, sep='\t')
lista = armo_entregas_desde_probs(df_, modelos=1, semillas=15)


# %%

guardo_en_archivos(lista, experimento='SC-0034-base-muchosmeses')
