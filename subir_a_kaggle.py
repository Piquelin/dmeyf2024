# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:52:37 2024

@author: jfgonzalez
"""

import re
import os

path_exp = 'C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/exp/'
experimento = 'KA421_4_1/'

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

# Uso de la función
ruta_archivo = path_exp + experimento + "modelo.txt"
params = extraer_parametros(ruta_archivo)



# %%
# Voy a mandar con un mensaje que incluya los parámetros usados y la cantidad de envíos

mensaje="lgbm entrenado en 4 meses abril predicho en junio"

experimento = "KA421_4_1/"

# Envío a la competencia

competencia="dm-ey-f-2024-primera"

# %%

for entregas in range (5000, 20001, 500):
    archivo = f"KA421_4_1_{entregas}.csv"
    path_archivo = path_exp + experimento + archivo
    print('Subiendo', archivo)
    !kaggle competitions submit -c {competencia} -f "{path_archivo}" -m ""

# %%




