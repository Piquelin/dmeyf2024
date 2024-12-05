# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:17:22 2024

@author: Piquelin
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm  # Para la barra de progreso


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
file = 'SC/expw_SC-0035_tb_future_prediccion.txt'
file = 'SC/expw227_SC-0006_tb_future_prediccion.txt'
file = 'SC/exp_03_SC-0001_tb_future_prediccion.txt'
file = 'vm_logs/SC/expw227_SC-0007_tb_future_prediccion.txt'
df_ = pd.read_csv(basepath + file, sep='\t')

# df_ = df_[df_['foto_mes'] == 202109]
lista = armo_entregas_desde_probs(df_, modelos=1, semillas=20)


# %% uno modelos


file = 'SC/expw_SC-0035_tb_future_prediccion.txt'
df_ = pd.read_csv(basepath + file, sep='\t')
lista = lista + armo_entregas_desde_probs(df_, modelos=1, semillas=15)


lista[0].index = lista[0].numero_de_cliente
lista[1].index = lista[1].numero_de_cliente
nuevo = pd.concat([lista[0].prom, lista[1].prom], axis=1)

l_nuevo = armo_entregas_desde_probs(df_, modelos=1, semillas=2)

# %%

guardo_en_archivos(lista, experimento='c03_entregas_SC-0007-SEMI')



# %% armo permanencia en top


def rank_and_accumulate_bins_polars(df, prob_col,
                                    foto_mes_col,
                                    cliente_col,
                                    num_bins):
    """
    Optimiza el ranking y cálculo acumulativo de bines utilizando Polars,
    con una barra de progreso para seguimiento.

    Args:
        df (pl.DataFrame or pd.DataFrame): DataFrame de entrada.
        prob_col (str): Nombre de la columna de probabilidades.
        foto_mes_col (str): Nombre de la columna que identifica los períodos.
        cliente_col (str): Nombre de la columna que identifica a los clientes.
        num_bins (int): Número de bines para dividir las probabilidades.

    Returns:
        pl.DataFrame: DataFrame con columnas de ranking
        y acumulativos por bines.
    """
    # Convertir a Polars si el DataFrame es Pandas
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # Agregar el tamaño del grupo para cada foto_mes
    df = df.with_columns(
        pl.col(foto_mes_col)
        .count()
        .over(foto_mes_col)
        .alias("foto_mes_size")
    )

    # Crear los bines por foto_mes
    df = df.with_columns(
        (pl.col(prob_col)
         .rank(descending=True, method="ordinal")
         .over(foto_mes_col) / pl.col("foto_mes_size") * num_bins
         ).cast(pl.Int32)  # Asegurar que los bines sean enteros
        .alias("bin")
         )

    # Crear columnas acumulativas por cliente con barra de progreso
    bin_columns = []
    for bin_num in tqdm(range(0, num_bins), desc="Calculando acumulativos"):
        bin_col_name = f"bin_{bin_num}_count"
        bin_columns.append(bin_col_name)
        df = df.with_columns(
            (pl.col("bin") == bin_num)
            .cast(pl.Int32)  # Asegurar que sea del tipo correcto
            .cum_sum()
            .over(cliente_col)
            .alias(bin_col_name)
            )

    return df


def combino_datos_competencia(result):
    # acá agregar las columnas al archivo
    competencia_df = pl.read_parquet("../datasets/competencia_03.parquet")

    result_df = competencia_df.join(
        result,
        on=["numero_de_cliente", "foto_mes"],  # Columnas comunes para el join
        how="left"  # Tipo de join: 'left', 'inner', 'outer', etc.
    )
    return result_df


def guardo_gzip(result_df):
    import gzip
    from datetime import datetime

    start = datetime.now()
    with gzip.open("../datasets/competencia_03_ranks.csv.gz", 'wb') as f:
        result_df.write_csv(f)
    print(f"gz saved at {datetime.now()-start}")


def bineo():
    basepath = '../exp/vm_logs/'
    file = 'SC/expw_SC-0036_tb_future_prediccion.txt'

    df_polars = pl.read_csv(basepath + file, separator='\t',
                            infer_schema_length=10000)

    # Aplicar la función
    result = rank_and_accumulate_bins_polars(
        df=df_polars,
        prob_col='m_2_1',
        foto_mes_col='foto_mes',
        cliente_col='numero_de_cliente',
        num_bins=11
    )

    result_df = combino_datos_competencia(result)
    return result_df

# %%
# bineo()
# guardo_gzip(result_df)
