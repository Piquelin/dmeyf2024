# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:07:07 2024

@author: jfgonzalez
"""

import os
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# %% FUNCIONES

def plot_ganancia_x_linea_de_corte(ganancia_acumulada, experimento):
    # Función para formatear el eje Y dividiendo por millones y mostrando tres decimales
    def millions_formatter(x, pos):
        return f'{x / 1_000_000:.3f}M'

    # Función para formatear el eje X con separador de miles
    def thousands_formatter(x, pos):
        return f'{x:,.0f}'

    if experimento[2] != 'prediccion.txt':
        serie = f'{experimento[1]}, mod_sem: {experimento[2][11:-4]}'
    else:
        serie = f'{experimento[1]}'

    # Graficar la suma acumulativa
    plt.figure(figsize=(10, 5))
    plt.plot(ganancia_acumulada, label=f"{serie}", color='b', linewidth=1)

    # Aplicar el formateador de millones en el eje Y
    plt.gca().yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    # Aplicar el formateador de miles en el eje X
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Establecer divisiones mayores cada 1000 en el eje X
    plt.gca().xaxis.set_major_locator(MultipleLocator(5000))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1000))
    plt.minorticks_on()

    # Encontrar el valor máximo de la ganancia acumulada
    max_y = ganancia_acumulada.max()
    max_x = ganancia_acumulada.arg_max()  # Índice del valor máximo

    # Limitar el rango de los ejes
    plt.xlim([0, 25000])
    plt.ylim([0, 120000000])

    # Título y etiquetas
    plt.title("Ganancia por línea de corte")
    plt.xlabel("Corte")
    plt.ylabel("Ganancia")

    # Dibujar líneas horizontales y verticales
    plt.axhline(max_y, color='g', linestyle='--', label=f'y={max_y:,d} ', linewidth=1)
    plt.axvline(max_x, color='r', linestyle='--', label=f'x={max_x:,d}', linewidth=1)

    # Añadir leyenda
    plt.legend(fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mostrar la gráfica
    plt.show()
    plt.close('all')


def cargo_mes_testing(mes_test='202106'):
    # Cargar solo las columnas necesarias del archivo .parquet
    df_real = pl.read_parquet(
        "./datasets/competencia_02.parquet",
        columns=["numero_de_cliente", "clase_ternaria", "foto_mes"])

    # Filtrar por foto_mes == 202106
    filtered_df = df_real.filter(pl.col("foto_mes") == mes_test).select([
        "numero_de_cliente",
        "clase_ternaria"])

    filtered_df = filtered_df.with_columns(
        pl.when(pl.col("clase_ternaria") == "BAJA+2")
        .then(273000)
        .otherwise(-7000)
        .alias("ganancia")
    )
    return filtered_df


def cargo_probabilidades_y_calculo_ganancia(df_test, experimento):

    try:
        # Intentar leer el archivo CSV
        df_prob = pl.read_csv(
            f'./exp/{experimento[1]}/{experimento[2]}',
            separator='\t',
            schema_overrides=[pl.Int64, pl.Int64, pl.Float64]
        ).select(["numero_de_cliente", "prob"])

        # print("Archivo cargado exitosamente.")

    except FileNotFoundError:
        print(f"Error: El archivo './exp/{experimento}/prediccion.txt' no se encontró.")
    except Exception as e:
        print(f"Se produjo un error: {e}")


    # Realizar la unión (join) usando "numero_de_cliente" como clave
    joined_df = df_test.join(
        df_prob,
        on="numero_de_cliente",  # Clave de la unión
        how="inner"  # Tipo de unión, puede ser "inner", "left", "outer", etc.
    )
    joined_df = joined_df.sort('prob', descending=True)

    # Calcular la suma acumulativa de la columna "ganancia"
    joined_df = joined_df.with_columns( pl.col("ganancia").cum_sum().alias("ganancia_acumulada"))

    return joined_df


def buscar_archivos_predic_txt(directorio_base):
    rutas_encontradas = []

    # Recorrer el directorio y las subcarpetas
    for ruta_directorio, subdirectorios, archivos in os.walk(directorio_base):
        for archivo in archivos:
            if archivo.startswith('predic') and archivo.endswith('.txt'):
                ruta = os.path.join(ruta_directorio, archivo).split('\\')
                rutas_encontradas.append(ruta)

    return rutas_encontradas


# %% MAIN

df_test = cargo_mes_testing(mes_test=202106)

directorio_base = 'C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp\\'
# directorio_base = 'E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas/exp'
archivos_exp = buscar_archivos_predic_txt(directorio_base)
# # %%
# for i, arch in enumerate(archivos_exp):
#     print(i, arch[1])

# archivos_exp = archivos_exp[30:59]

# %%
for exp in archivos_exp:
    df_gan = cargo_probabilidades_y_calculo_ganancia(df_test, experimento = exp)
    plot_ganancia_x_linea_de_corte(df_gan['ganancia_acumulada'], experimento = exp)
