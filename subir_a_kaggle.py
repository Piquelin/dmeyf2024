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

# %%

# defino ruta de los archivos del experimento
path_exp = 'C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/'
path_exp = 'E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas/exp/'
experimento = 'KA4215_06/'

# # Levanto parametros del log para este experimento
# ruta_archivo = path_exp + experimento + "modelo.txt"
# params = extraer_parametros(ruta_archivo)

# mensajes y detalles para sibir a kaggle
mensaje= f"cuarta_sem_clust" #" Parametros: {params}"
competencia="dm-ey-f-2024-primera"

# experimento[0:-1]

# %% Quiero optimizar la entrega entre 1000 y 10000
# más ganacia con menos envíos mejor


for entregas in range (7000, 14001, 500):
    archivo = f"{experimento[0:-1]}_{entregas}.csv"
    path_archivo = path_exp + experimento + archivo
    print('Subiendo', archivo)
    !kaggle competitions submit -c {competencia} -f "{path_archivo}" -m "{mensaje}"
    time.sleep(1.3) # Seconds


# %%

# Inicializar la API usando las credenciales de kaggle.json
api = KaggleApi()
api.authenticate()

# %%

all_submissions = []
page = 1
page_size = 100  # Max number of submissions per page
max_pages = 20 # Max number of pages to fetch

while True:
    # Get the submissions for the current page
    submissions = api.competition_submissions(competencia, page_token=page, page_size=page_size)

    # If no submissions are returned, we have reached the last page
    if len(submissions) == 0 or page > max_pages:
        break

    # Append the submissions data
    all_submissions.extend([{
        'submission_id': sub.ref,
        'date': sub.date,
        'score': sub.publicScore,
        'description': sub.description,
        'fileName': sub.fileName,
        'submittedBy': sub.submittedBy
    } for sub in submissions])

    # Move to the next page
    page += 1

# Convert the list into a DataFrame for better readability and manipulation
df = pd.DataFrame(all_submissions)

# %%
# Optionally, save the scores to a CSV file
df.to_csv('./scores/my_kaggle_submissions.csv', index=False)

print("Submission scores saved to 'my_kaggle_submissions.csv'")


# %%
# Get your submission history for the competition
submissions = api.competition_submissions(competencia, page_token=2, page_size=20)

# Create a list of dictionaries to store submission details
submission_list = []

for submission in submissions:
    submission_info = {
        'SubmissionId': submission.ref,
        'FileName': submission.fileName,
        'Date': submission.date,
        'Score': submission.publicScore,
        'Description': submission.description
    }
    submission_list.append(submission_info)

# Convert the list into a DataFrame for better readability and manipulation
df = pd.DataFrame(submission_list)


int(df.Score)
