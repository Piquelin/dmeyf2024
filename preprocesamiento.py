# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:06:29 2024

@author: jfgonzalez
"""

import polars as pl
import pandas as pd
import re


# %%

workin_dir =('C:/Users/jfgonzalez/Documents/documentación_maestría/' +
             'economía_y_finanzas/')
archivo_datos = './datasets/competencia_02.parquet'


competencia_02 = pl.read_parquet(archivo_datos)


#%%


def obtener_campos_monetarios(dataset):
    # Obtener los nombres de las columnas del dataset
    columnas = dataset.columns  # o dataset if ya es una lista de columnas
    
    # Definir la expresión regular
    pattern = r"^(m|Visa_m|Master_m|vm_m)"
    
    # Filtrar las columnas que coinciden con la expresión regular
    campos_monetarios = [col for col in columnas if re.match(pattern, col)]
    
    return campos_monetarios


def encuentro_ceros(df): # AKA AsignarNA_campomeses (catástrofe)
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
        df_mes = df_mes.with_columns([pl.when(pl.col(col).is_not_null()).then(None).otherwise(pl.col(col)).alias(col) for col in cols_cero])
        
        print('del mes', mes, 'reemplazamos las columnas', cols_cero)
        listadfs.append(df_mes)
    
    # Concatenar todos los dataframes de meses
    resultado_final = pl.concat(listadfs)
    return resultado_final

# acá ya tengo el df original con los campos todo cero como nulos
competencia_02 = encuentro_ceros(competencia_02)

# %% listas varias

def busco_inflacion(ultimo_mes='10_24', inicio="2019-01-01", fin="2021-08-01"):
    '''
    ultimo mes: MM_AA
    '''
    PATH = ('https://www.indec.gob.ar/ftp/cuadros/economia/' +
            f'sh_ipc_{ultimo_mes}.xls')
    
    ipc_nac = pl.read_excel(PATH, sheet_name='Variación mensual IPC Nacional',
                            read_options={"skip_rows": 0, "header_row":4,
                                          "n_rows":30, } )
   
    new_columns = ipc_nac.row(0)  # primera fila como encabezado
    # ipc_nac = ipc_nac.slice(2)  # eliminar la primera fila
    
    meses = pl.DataFrame(pl.Series(new_columns[1:])
                         .str.strptime(pl.Date, "%F %T", strict=False))
    
    ipc_df = pl.concat([meses, ipc_nac[3,1:].transpose()], how='horizontal')
    ipc_df.columns = (['mes', 'IPC'])

    
    
    # Filtrar el rango de fechas
    ipc_df = ipc_df.filter(
        (pl.col('mes') >= pl.lit(inicio).str.strptime(pl.Date, "%Y-%m-%d")) & 
        (ipc_df['mes'] <= pl.lit(fin).str.strptime(pl.Date, "%Y-%m-%d"))
    )
    
    
    return ipc_df

busco_inflacion()

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



# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))



