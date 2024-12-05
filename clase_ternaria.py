# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:58:02 2024

@author: jfgonzalez
"""


import polars as pl
import gzip
import pandas as pd

# dataset = pl.read_csv("datasets/competencia_01_crudo.csv", infer_schema_length=10000)
# dataset = pl.read_csv("../datasets/competencia_02_crudo.csv.gz", infer_schema_length=500000)
dataset = pl.read_csv("../datasets/competencia_03_crudo.csv.gz", infer_schema_length=500000)


#%% pivoteo

# pivoteo con las fechas para compara los meses. tomo el valor de active _quarter para simplificar los reemplazos
tabla = dataset[['numero_de_cliente', 'foto_mes', 'active_quarter']].pivot(on='foto_mes', index='numero_de_cliente', values='active_quarter')


# Reemplazar valores nulos por False y valores enteros por True
tabla_bool = tabla.with_columns([
    pl.col(column).is_not_null().alias(column)
    for column in tabla.columns[1:]
])


#%% funcion de evaluación

# con esto evalúo cada uno de los primeros 4 meses en función a los dos siguientes
# para el caso de clientes intermitentes se los considera como que se fueron en el mes en cuestión
# y se evalúa nuevamente a partir que se reincorporan.

def evaluar_condiciones(tabla: pl.DataFrame, col1: str, col2: str, col3: str) -> pl.Expr:

    evaluacion = (
        pl.when(pl.col(col1) == False)
        .then(None)
        .when(
            (pl.col(col2) == False)
        )
        .then(pl.lit("BAJA+1"))
        .when(
            (pl.col(col3) == True)
        )
        .then(pl.lit("CONTINUA"))
        .when(
            (pl.col(col3) == False)
        )
        .then(pl.lit("BAJA+2"))

        .otherwise(None)
    )

    return evaluacion

#%% Genero los resultados para los 4 primeros meses

lista_meses = tabla.columns[1:]
meses_completos = len(lista_meses)-2

for i in range(meses_completos):
    tabla_bool = tabla_bool.with_columns(evaluar_condiciones(tabla_bool, lista_meses[i], lista_meses[i+1], lista_meses[i+2]).alias(f"evaluacion_{lista_meses[i]}"))


#%% unpivoteo y hago join

clase_cliente = tabla_bool.unpivot(index='numero_de_cliente')
clase_cliente = clase_cliente.rename({
    "variable": "foto_mes",
    'value':'clase_ternaria'})
# descarto columas auxiliares
clase_cliente = clase_cliente.filter(pl.col("foto_mes").str.starts_with("e"))
# paso en limpio las fechas
clase_cliente = clase_cliente.with_columns(pl.col("foto_mes").str.slice(-6).cast(pl.Int64).alias("foto_mes"))

# %%  Realizar la unión usando 'numero_de' y 'foto_mes' como claves
resultado = dataset.join(
    clase_cliente,
    on=["numero_de_cliente", "foto_mes"],
    how="left"
)


# %% guardo


# resultado.write_csv('../datasets/competencia_03.csv', separator=",")
resultado.write_parquet("../datasets/competencia_03.parquet")

file_path = "../datasets/competencia_03.csv.gz"
with gzip.open(file_path, 'wb') as f:
    resultado.lazy().collect().write_csv(f, separator=",")


# %% clases_por_mes

clases_por_mes = []
for ev_mes in tabla_bool.columns[-meses_completos:]:
    valores = tabla_bool[ev_mes].value_counts().to_pandas()
    valores.set_index(valores[ev_mes], inplace=True)
    valores.drop(ev_mes, axis=1, inplace=True)
    valores.index.rename('clase', inplace=True)
    valores.columns = [ev_mes]

    # print(valores)
    clases_por_mes.append(valores)

tabla_meses = pd.concat(clases_por_mes, axis=1).T
print(tabla_meses)

# %%

# res_sin_prestamos = resultado.drop(['cprestamos_personales', 'mprestamos_personales'])
# res_sin_prestamos.write_csv("resultado_sin_prestamos.csv", separator=",")
