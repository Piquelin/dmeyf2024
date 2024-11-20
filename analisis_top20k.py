# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:23:38 2024

@author: jfgonzalez
"""

import pandas as pd

import numpy as np
import os
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# %% FUNCIONES

def plot_ganancia_x_linea_de_corte(ganancia_acumulada, gan_2=pd.Series('nada'), serie='Ganacia acum'):
    # Función para formatear el eje Y dividiendo por millones y mostrando tres decimales
    def millions_formatter(x, pos):
        return f'{x / 1_000_000:.3f}M'

    # Función para formatear el eje X con separador de miles
    def thousands_formatter(x, pos):
        return f'{x:,.0f}'

    
    # Graficar la suma acumulativa
    plt.figure(figsize=(18, 9))
    plt.plot(ganancia_acumulada, label=f"{serie}", color='b', linewidth=1)
    
    if gan_2.dtype == 'int64':
        plt.plot(gan_2, label="Top 20k", color='r', linewidth=1)

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
    max_x = ganancia_acumulada.idxmax()  # Índice del valor máximo

    # Limitar el rango de los ejes
    plt.xlim([0, 20500])
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
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Mostrar la gráfica
    plt.show()
    plt.close('all')


def armo_entregas_desde_probs(df_, modelos=3, semillas=20, entregas=False, calc_gan=False):
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
    
    if calc_gan:
        for modelo in lista_prob_prom:
            modelo["ganancia"] = modelo["clase_ternaria"].apply(
                lambda x: 273000 if x == "BAJA+2" else -7000)
            modelo['ganancia_acumulada'] = modelo['ganancia'].cumsum()
    
    return lista_prob_prom

# %%


PATH = 'C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/'
FILE = 'exp/vm_logs/SC-0020_pollo_parrillero_future_prediccion.txt'
FILE = 'exp/vm_logs/SC-0011_tb_future_prediccion.txt'
FILE2 = 'exp/vm_logs/SC-0029_tb_future_prediccion.txt'

score_full = pd.read_csv(PATH+FILE, sep='\t')
score_top20k = pd.read_csv(PATH+FILE2, sep='\t')



lista_prom = armo_entregas_desde_probs(score_full, modelos=3, semillas=5,
                                       entregas=False, calc_gan=True)


lista_top20k = armo_entregas_desde_probs(score_top20k, modelos=3, semillas=20,
                                         entregas=False, calc_gan=True)

# %%

plot_ganancia_x_linea_de_corte(lista_prom[2]['ganancia_acumulada'], gan_2=lista_top20k[2]['ganancia_acumulada'],  serie='prom_mod_base')


plot_ganancia_x_linea_de_corte(lista_top20k[2]['ganancia_acumulada'], serie='Top 20k')
