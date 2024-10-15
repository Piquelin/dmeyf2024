# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:07:07 2024

@author: jfgonzalez
"""

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

    # Graficar la suma acumulativa
    plt.figure(figsize=(7, 3))
    plt.plot(ganancia_acumulada, label=f"{experimento}", color='b', linewidth=1)

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
    plt.ylim([0, max_y * 1.1])

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
    filtered_df = df_real.filter(pl.col("foto_mes") == 202106).select([
        "numero_de_cliente",
        "clase_ternaria"])

    filtered_df = filtered_df.with_columns(
        pl.when(pl.col("clase_ternaria") == "BAJA+2")
        .then(273000)
        .otherwise(-7000)
        .alias("ganancia")
    )
    return filtered_df


def cargo_probabilidades_y_calculo_ganancia(df_test, experimento = 'KA7250_us_25'):

    try:
        # Intentar leer el archivo CSV
        df_prob = pl.read_csv(
            f'./exp/{experimento}/prediccion.txt',
            separator='\t',
            dtypes=[pl.Int64, pl.Int64, pl.Float64]
        ).select(["numero_de_cliente", "prob"])

        print("Archivo cargado exitosamente.")

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



# %%

df_test = cargo_mes_testing(mes_test='202106')
df_gan = cargo_probabilidades_y_calculo_ganancia(df_test, experimento = 'KA7250_us_25')
plot_ganancia_x_linea_de_corte(df_gan['ganancia_acumulada'], experimento = 'KA7250_us_25')

# %%

import os

# Función para buscar archivos que empiecen con 'predic' y terminen con '.txt'
def buscar_archivos_predic_txt(directorio_base):
    rutas_encontradas = []

    # Recorrer el directorio y las subcarpetas
    for ruta_directorio, subdirectorios, archivos in os.walk(directorio_base):
        for archivo in archivos:
            if archivo.startswith('predic') and archivo.endswith('.txt'):
                rutas_encontradas.append(os.path.join(ruta_directorio, archivo))

    return rutas_encontradas

# Aún no se ha especificado el directorio
directorio_base = '/ruta/al/directorio'  # Reemplazar con la ruta real
# buscar_archivos_predic_txt(directorio_base) # Este comando buscaría en el directorio

# Por ahora solo mostramos la función para explorar la carpeta
"Función lista para usarse una vez que se especifique el directorio"
