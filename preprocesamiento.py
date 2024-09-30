# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:06:29 2024

@author: jfgonzalez
"""

import polars as pl
import re


# %%

workin_dir =('C:/Users/jfgonzalez/Documents/documentación_maestría/' +
             'economía_y_finanzas/')
archivo_datos = './datasets/competencia_01.csv'

# %% 

resultado = pl.read_csv(archivo_datos)

# %%



#%%

def encuentro_ceros(df):
    # Obtiene la lista de meses únicos
    meses = df['foto_mes'].unique().to_list()
    
    # Recorre los meses
    listadfs = []
    for mes in meses:
        df_mes = df.filter(pl.col("foto_mes") == mes)
        
        # Verifica qué columnas tienen todos ceros excepto 'foto_mes' y 'clase_ternaria'
        todo_cero_mask = df_mes.select(pl.all().exclude("foto_mes", "clase_ternaria").cast(pl.Float64).sum() == 0).row(0)
        todo_cero_mask = (False,) + todo_cero_mask + (False,)
        cols_cero = [col for col, is_cero in zip(df_mes.columns, todo_cero_mask) if is_cero]
        
        # Asigna None a esas columnas
        df_mes = df_mes.with_columns([pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col) for col in cols_cero])
        
        print('del mes', mes, 'reemplazamos las columnas', cols_cero)
        listadfs.append(df_mes)
    
    # Concatenar todos los dataframes de meses
    resultado_final = pl.concat(listadfs)
    return resultado_final

# Llamada de ejemplo:
resultado = encuentro_ceros(resultado)

# %%

# valores financieros
# meses que me interesan
vfoto_mes = [202101, 202102, 202103, 202104, 202105, 202106]

# los valores que siguen fueron calculados por alumnos
#  si no esta de acuerdo, cambielos por los suyos

# momento 1.0  31-dic-2020 a las 23:59
vIPC [ 0.9680542110, 0.9344152616, 0.8882274350,
      0.8532444140, 0.8251880213, 0.8003763543]

vdolar_blue = [157.900000, 149.380952, 143.615385,
               146.250000, 153.550000, 162.000000]


vdolar_oficial = [ 91.474000,  93.997778,  96.635909,
                  98.526000,  99.613158, 100.619048]
  
vUVA = [  0.9669867858358365, 0.9323750098728378, 0.8958202912590305,
        0.8631993702994263, 0.8253893405524657, 0.7928918905364516 ]

# %%



def obtener_campos_monetarios(dataset):
    # Obtener los nombres de las columnas del dataset
    columnas = dataset.columns  # o dataset if ya es una lista de columnas
    
    # Definir la expresión regular
    pattern = r"^(m|Visa_m|Master_m|vm_m)"
    
    # Filtrar las columnas que coinciden con la expresión regular
    campos_monetarios = [col for col in columnas if re.match(pattern, col)]
    
    return campos_monetarios

# Llamada de ejemplo:
# campos_monetarios = obtener_campos_monetarios(dataset)

obtener_campos_monetarios(resultado)
