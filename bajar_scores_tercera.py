# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:57:10 2024

@author: Piquelin
"""

import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi


# %% Funciones

def authenticate_kaggle_api():
    """Autentica la API de Kaggle y devuelve la instancia."""
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_kaggle_submissions(api, competition):
    """
    Descarga las submissions de una competencia de Kaggle.

    Args:
        api (KaggleApi): Instancia autenticada de KaggleApi.
        competition (str): Nombre de la competencia.

    Returns:
        pd.DataFrame: DataFrame con las submissions.
    """
    all_submissions = []
    submissions = api.competition_submissions(competition)

    all_submissions.extend([{
        'submission_id': sub.ref,
        'date': sub.date,
        'score': sub.publicScore,
        'description': sub.description,
        'fileName': sub.fileName,
        'submittedBy': sub.submittedBy
    } for sub in submissions])

    return pd.DataFrame(all_submissions)


def combine_and_update_scores(new_scores, scores_file):
    """
    Combina un DataFrame de nuevos scores con un archivo existente,
    elimina duplicados y guarda el resultado.

    Args:
        new_scores (pd.DataFrame): DataFrame con los nuevos scores.
        scores_file (str): Ruta al archivo CSV de scores.
    """
    if os.path.exists(scores_file):
        # Cargar el archivo existente
        df_existing = pd.read_csv(scores_file)
        # Combinar con los nuevos scores
        df_combined = pd.concat([df_existing, new_scores], ignore_index=True)
    else:
        # Si el archivo no existe, usar solo los nuevos scores
        df_combined = new_scores

    # Eliminar duplicados basándose en columnas clave
    df_combined = df_combined.drop_duplicates(subset=['submission_id'],
                                              keep='last')

    # Guardar el DataFrame combinado en el archivo
    df_combined.to_csv(scores_file, index=False)
    print(f"Scores actualizados guardados en {scores_file}.")


def main():
    """Función principal para ejecutar el flujo completo."""
    # Configurar la competencia y el archivo de salida
    competencia = "dm-ey-f-2024-tercera"
    scores_file = "scores_tercera.csv"

    # Autenticar Kaggle API
    api = authenticate_kaggle_api()

    # Obtener los nuevos scores de Kaggle
    new_scores = fetch_kaggle_submissions(api, competencia)

    # Combinar con el archivo existente y guardar los resultados
    combine_and_update_scores(new_scores, scores_file)


if __name__ == "__main__":
    main()
