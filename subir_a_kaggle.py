# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:52:37 2024

@author: jfgonzalez
"""

import time
import re


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
experimento = 'KA7250/'

# Levanto parametros del log para este experimento
ruta_archivo = path_exp + experimento + "modelo.txt"
params = extraer_parametros(ruta_archivo)

# mensajes y detalles para sibir a kaggle
mensaje= f"del pipeline del jueves, como prueba. Parametros: {params}"
competencia="dm-ey-f-2024-primera"

experimento[0:-1]

# %%

for entregas in range (8000, 15001, 500):
    archivo = f"{experimento[0:-1]}_{entregas}.csv"
    path_archivo = path_exp + experimento + archivo
    print('Subiendo', archivo)
    !kaggle competitions submit -c {competencia} -f "{path_archivo}" -m "{mensaje}"
    time.sleep(1.3) # Seconds


# %%




