# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:33:35 2024

@author: jfgonzalez
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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



def grafico_3m_202106(df_graf, titulo='Pollo-parrillero'):
    # Crear figura y subplots
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
    
    # Configuración del rango de ejes y formato de unidades
    for ax in axs:
        ax.set_xlim(8000, 16000)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1e3):,d}K"))  # Separador de miles en eje X
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y / 1e6:.0f}M"))  # Millones con 3 decimales en eje Y
        
        
        # Configuración de la grilla
        ax.grid(visible=True, which='major', axis='y', linestyle='-', linewidth=0.8)
        ax.grid(visible=True, which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5e6))  # Divisiones mayores cada 5 millones
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1e6))  # Divisiones menores cada 1 millón
        ax.grid(visible=True, which='major', axis='x', linestyle='--', linewidth=0.8, alpha=0.5)
    
    
    # Primer subplot
    axs[0].plot(df_graf.iloc[:, 0:20], c='grey', alpha=0.5, linewidth=0.5)
    axs[0].plot(df_graf.iloc[:, 61], c='red', alpha=1, linewidth=1)
    
    # Segundo subplot
    axs[1].plot(df_graf.iloc[:, 20:40], c='grey', alpha=0.5, linewidth=0.5)
    axs[1].plot(df_graf.iloc[:, 62], c='red', alpha=1, linewidth=1)
    
    # Tercer subplot
    axs[2].plot(df_graf.iloc[:, 40:60], c='grey', alpha=0.5, linewidth=0.5)
    axs[2].plot(df_graf.iloc[:, 63], c='red', alpha=1, linewidth=1)
    
    fig.suptitle(titulo)
    
    # Ajustes finales
    plt.tight_layout()
    plt.show()
    plt.close()
    
    

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
    


def armo_entregas_desde_probs(df_, modelos=3, semillas=20):
    lista_prob_prom = []
    
    for i in range(modelos):
        df_entrega =  df_[['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
        # df_entrega['numero_de_cliente'] = df_['numero_de_cliente']
        df_entrega['prom'] = df_.iloc[:,(i*semillas + 3):(i*semillas + 3 + semillas)].T.mean()
        df_entrega = df_entrega.sort_values('prom', ascending=False).reset_index()
        df_entrega = df_entrega[['numero_de_cliente',  'foto_mes', 'clase_ternaria', 'prom']]
        lista_prob_prom.append(df_entrega)
    
    
    total_clientes =len(lista_prob_prom[0])
    for i in range(modelos):
        for corte in range(8000, 16001, 500):
            array = np.zeros((total_clientes, 1))
            array[:corte] = 1
            lista_prob_prom[i][f'pred_{corte}'] = array.astype(int)
    
    return lista_prob_prom
            


def guardo_en_archivos(dfs, experimento):
    # Crear directorio de entregas
    directorio_entregas = f'entregas_{experimento}'
    os.makedirs(directorio_entregas, exist_ok=True)
    
    # Guardar predicciones de cada modelo en archivos CSV
    for modelo, df in enumerate(dfs):
        for col in df.columns:
            if col.startswith('pred_'):
                corte = col.split('_')[1]  # Extraer el número del corte
                df_pred = df[['numero_de_cliente', col]].copy()
                df_pred.columns = ['numero_de_cliente', 'Predicted']  # Renombrar columnas
    
                # Nombre de archivo y ruta de guardado
                archivo_nombre = f"{experimento}_{modelo}_{corte}.csv"
                archivo_ruta = os.path.join(directorio_entregas, archivo_nombre)
                
                # Guardar en CSV
                df_pred.to_csv(archivo_ruta, index=False)
    return None


# %%



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
# basepath = 'E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas/exp/vm_logs/'
file='SC-0020_pollo_parrillero_future_prediccion.txt'
file = 'SC-0024_ipp-bagg-mm-06_future_prediccion.txt'


ganancias = calcular_cortes_y_promedios(basepath, file)

#%%

grafico_3m_202106(df_graf=ganancias.loc[8000:16000], titulo='ipp-bagg-mm')

# %%


basepath ='C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/vm_logs/'
file1 = 'SC-0021_pollo_bagg_ka_future_prediccion.txt'
# file0 = 'SC-0020_pollo_parrillero_future_prediccion.txt'
file2 = 'SC-0023_pollo_bagg_dart_ka_future_prediccion.txt'
file3 = 'SC-0025_ipp_bagg_mm_ka_future_prediccion.txt'
file4 = 'SC-0026_ipp-bagg-dart-mm_future_prediccion.txt'

# df0 = pd.read_csv(basepath+file0, sep='\t')
df1 = pd.read_csv(basepath+file1, sep='\t')
df2 = pd.read_csv(basepath+file2, sep='\t')
df3 = pd.read_csv(basepath+file3, sep='\t')    
df4 = pd.read_csv(basepath+file4, sep='\t')

df_all = pd.concat([df1, df2.iloc[:,3:], df3.iloc[:,3:], df4.iloc[:,3:]], axis=1, )

    
# lista = armo_entregas_desde_probs(df_, modelos=3, semillas=20)
lista = armo_entregas_desde_probs(df_all, modelos=1, semillas=240)


# %%

guardo_en_archivos(lista, experimento='SC-0021-23-25-26_promedio')



# %%
# len(lista[0])
# df_unos = lista[0][['numero_de_cliente', 'pred_16000']]
# df_unos['Predicted'] = 1
# df_unos = df_unos[['numero_de_cliente', 'Predicted']]
# df_unos.to_csv('todouno_comp_02', index=False)

# de acá sale que hay unos 1033 BAJA+2 y la ganacia máxima es 282.009
# andamos entre los 600 y 700 aciertos


# %%


    


# %%


# # mensajes y detalles para subir a kaggle
# mensaje= "promedios_primera (KA-0001_01_056_s512977)"

# # 1 y 2
# modelo = 4
# # 484751   641909   212561
# # 582781   536453   525773
# semilla = 525773


# # %% bajo valores de los submits limite últimos 50

# contador_entregas = 0
# for entregas in range (8000, 13001, 1000):
#     archivo = f"{experimento[0:-1]}_{modelo}_{semilla}_{entregas}.csv"
#     path_archivo = path_exp + experimento + archivo
#     print('Subiendo', archivo)
#     !kaggle competitions submit -c {competencia} -f "{path_archivo}" -m "{mensaje}"
#     time.sleep(1.3)  # Seconds
#     contador_entregas = contador_entregas + 1

# %%


df_errores = lista[0][['numero_de_cliente', 'foto_mes', 'clase_ternaria', 'prom', 'pred_9000']]

# Crear la columna 'err' basada en las condiciones dadas
df_errores['err'] = np.where(
    (df_errores['pred_9000'] == 1) & (df_errores['clase_ternaria'] == 'CONTINUA'), 1,
    np.where(
        (df_errores['pred_9000'] == 0) & (df_errores['clase_ternaria'] != 'CONTINUA'), 1,
        0
    )
)

df_errores[df_errores['err']==1]



    
