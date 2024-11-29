# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:27:53 2024

@author: Piquelin
"""




import lightgbm as lgb



local_filename = 'exp/models/modelo_02_037_s417019.model'
dataset_filename = 'datasets/competencia_03.parquet'

# Cargar el modelo con LightGBM
model = lgb.Booster(model_file=local_filename)

print("Modelo cargado con éxito")

# %%


model.feature_importance()
