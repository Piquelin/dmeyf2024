# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:52:37 2024

@author: jfgonzalez
"""

import time
import re
import os
from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd

# %%

# Inicializar la API usando las credenciales de kaggle.json
api = KaggleApi()
api.authenticate()

# defino ruta de los archivos del experimento
path_exp = 'C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/'
path_exp = 'E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas/exp/'
competencia = "dm-ey-f-2024-primera"

# %% Cargo datos a mano porque me a más flexibilidad

experimento = 'KA7455_quant/'

# mensajes y detalles para subir a kaggle
mensaje= f"PP7235_25_s1_sin_pres -('num_grad_quant_bins', lower = 2L, upper = 50L)"

# 1 y 2
modelo = 4
# 484751   641909   212561
# 582781   536453   525773
semilla = 525773


# %% bajo valores de los submits limite últimos 50

contador_entregas = 0
for entregas in range (8000, 13001, 1000):
    archivo = f"{experimento[0:-1]}_{modelo}_{semilla}_{entregas}.csv"
    path_archivo = path_exp + experimento + archivo
    print('Subiendo', archivo)
    !kaggle competitions submit -c {competencia} -f "{path_archivo}" -m "{mensaje}"
    time.sleep(1.3)  # Seconds
    contador_entregas = contador_entregas + 1


# %%

all_submissions = []
submissions = api.competition_submissions(competencia)

all_submissions.extend([{
    'submission_id': sub.ref,
    'date': sub.date,
    'score': sub.publicScore,
    'description': sub.description,
    'fileName': sub.fileName,
    'submittedBy': sub.submittedBy
} for sub in submissions])

# Convert the list into a DataFrame for better readability and manipulation
df = pd.DataFrame(all_submissions)

# %%
# guardo scores en la carpeta del experimento
df_ultima_subida = df[:contador_entregas]
direccion_score = f'{path_exp}{experimento}Scores_{experimento[0:-1]}_{modelo}_{semilla}.csv'
df_ultima_subida.to_csv(direccion_score, index=False)

print(df_ultima_subida[['fileName', 'score']])
print(f"guardado en  {direccion_score}")


# %%

# esto era para extraer paarametros del archvo del log
# experimentos de la familia "z4210"

# # Levanto parametros del log para este experimento
# ruta_archivo = path_exp + experimento + "modelo.txt"
# params = extraer_parametros(ruta_archivo)

def extraer_parametros(ruta_archivo):
    # Cargar el contenido del archivo de texto
    with open(ruta_archivo, "r") as file:
        content = file.read()

    # Encontrar la sección entre 'parameters:' y 'end of parameters'
    section_pattern = r'parameters:(.*)end of parameters'
    section_match = re.search(section_pattern, content, re.DOTALL)

    if section_match:
        parameters_section = section_match.group(1)

        # Usar una expresión regular para encontrar los parámetros solo en la sección
        param_pattern = r'\[([a-zA-Z_]+):\s*([^\]]+)\]'
        matches = re.findall(param_pattern, parameters_section)

        # Convertir las coincidencias en un diccionario
        parametros = {param: valor for param, valor in matches}

        # Construir un texto con la lista de parámetros
        texto_parametros = ";".join([f"{param}: {valor}" for param, valor in parametros.items()])
        return texto_parametros
    else:
        return "No se encontró la sección de parámetros."
