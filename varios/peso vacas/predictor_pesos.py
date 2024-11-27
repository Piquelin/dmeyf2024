# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:21:38 2024

@author: jfgonzalez
"""



#!pip install optuna 
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice, plot_contour

import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# %%


# df = pd.read_csv('./dataset_4peso.csv')

# %%

# columna_a_predecir = 'pesada_4'

def data_para_enrtenar(df, target):
    # elimino nulos del targuet y saco el índice
    data = df[~df[target].isna()]
    target = data.pop(target)
    data.index = data.camp_ide_concat
    data = data.drop(columns='camp_ide_concat')
    
    return data, target

# data, target = data_para_enrtenar(df, target=columna_a_predecir)


# %%
def optimizo_modelo(data, target):
    
    #  acá eligo que optimizar con la bayesiana
    def objective(trial,data=data, target=target):
        
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
        param = {
            
            'metric': 'rmse', 
            'random_state': 48,
            'n_estimators': 20000,
            'verbose': -1, 
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 500, 1000),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
            'num_leaves' : trial.suggest_int('num_leaves', 10, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        }
        model = LGBMRegressor(**param)  
        
        model.fit(train_x,train_y,eval_set=[(test_x,test_y)],
                  # early_stopping_rounds=100,
                  callbacks=[lgb.early_stopping(stopping_rounds=50),],
                  # verbose=False)
                  )
        
        preds = model.predict(test_x)
        
        rmse = root_mean_squared_error(test_y, preds) # ,squared=False)
        
        return rmse
    
    
    # creo estudio
    study = optuna.create_study(direction='minimize')
    
    # corro bayesiana
    study.optimize(objective, n_trials=50)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    
    # miro optimizacion
    plot_optimization_history(study)
    plot_param_importances(study)
    
    # parametros optimizados
    params=study.best_params   
    params['random_state'] = 48
    params['n_estimators'] = 20000 
    params['metric'] = 'rmse'
    params['verbose']=-1
    
    return params

# params = optimizo_modelo(data, target)

# %% entreno con los mejores parametros

def entreno_modelo_final(params, data, target):

   
    preds = np.zeros(data.shape[0])
    kf = KFold(n_splits=5,random_state=48,shuffle=True)
    rmse=[]  # list contains rmse for each fold
    n=0
    
    for trn_idx, test_idx in kf.split(data, target):
        X_tr,X_val=data.iloc[trn_idx],data.iloc[test_idx]
        y_tr,y_val=target.iloc[trn_idx],target.iloc[test_idx]
        model = LGBMRegressor(**params)
        model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(stopping_rounds=100),])
        preds+=model.predict(data)/kf.n_splits
        rmse.append(root_mean_squared_error(y_val, model.predict(X_val)))
        n+=1
    
    return model, preds, rmse

# model, preds, rmse = entreno_modelo_final(params, data=data, target=target)

# %% miro el modelo entrenado


def grafico_precision_modelo(target, preds, columna_a_predecir, rmse, valores_columna=None):
    """
    Genera un gráfico de dispersión para comparar las predicciones con los valores reales.
    
    Parámetros:
        target (array-like): Valores reales.
        preds (array-like): Valores predichos.
        columna_a_predecir (str): Nombre de la columna objetivo.
        rmse (float): Error cuadrático medio.
        valores_columna (array-like, opcional): Valores de una columna para colorear los puntos.
                                                Si contiene NaN, los puntos serán blancos.
    """
    # Calcular la desviación absoluta entre las predicciones y el target
    deviation = np.abs(target - preds)
    
    # Configurar colores para los puntos
    legend_labels = None
    if valores_columna is not None:
        # Verificar si la columna es categórica o continua
        valores_unicos = len(np.unique(valores_columna[~np.isnan(valores_columna)]))  # Ignorar NaN al contar únicos
        
        if valores_unicos < 20:  # Categórica
            categorias = np.unique(valores_columna[~np.isnan(valores_columna)])
            colores = []
            cmap = plt.get_cmap('tab20')  # Mapa de colores categóricos
            for i, valor in enumerate(valores_columna):
                if np.isnan(valor):
                    colores.append('white')
                else:
                    colores.append(cmap(np.where(categorias == valor)[0][0] / len(categorias)))
            legend_labels = {categoria: cmap(i / len(categorias)) for i, categoria in enumerate(categorias)}
        else:  # Continua
            # Normalizar entre 0 y 1 para cmap
            min_val = np.nanmin(valores_columna)
            max_val = np.nanmax(valores_columna)
            colores = (valores_columna - min_val) / (max_val - min_val)
            cmap = 'jet'
    else:
        # Usar la desviación absoluta como gradación de colores
        colores = deviation
        cmap = 'jet'
    
    # Crear el gráfico
    plt.figure(figsize=(10, 8))  # Ajustar el tamaño del gráfico
    scatter = plt.scatter(
        target, preds, c=colores, cmap=cmap, s=10, alpha=0.8, edgecolors='k'
    )
    
    # Agregar una línea de referencia
    plt.plot(
        [min(target), max(target)], [min(target), max(target)], 
        color='red', linestyle='--', linewidth=1
    )
    
    # Agregar una barra de color si es continua
    if valores_columna is not None and valores_unicos >= 20:
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'Valores normalizados de la columna seleccionada: {valores_columna.name}')
    elif valores_columna is not None and legend_labels is not None:
        # Crear leyenda para las categorías
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=str(cat))
                   for cat, color in legend_labels.items()]
        plt.legend(handles=handles, title=f"Categorías: {valores_columna.name}", loc='lower right', fontsize='small')
    else:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Desviación Absoluta')
    
    # Etiquetas y título
    plt.title(
        f'Comparación de Predicciones vs Real: {columna_a_predecir}\nError cuadrático medio: {np.mean(rmse):.2f}'
    )
    plt.xlabel('Peso real Kg')
    plt.ylabel('Predicciones Kg')
    

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()


# %%



df = pd.read_csv('./dataset_4gan.csv')
columna_a_predecir = 'gancia_pesada_4'

data, target = data_para_enrtenar(df, target=columna_a_predecir)

params = optimizo_modelo(data=data, target=target)

model, preds, rmse = entreno_modelo_final(params, data=data, target=target)


# pesos desconocidos
# datos_futuro = model.predict(X)
# datos_futuro.to_csv('gan4.csv')

# %%


df = pd.read_csv('./dataset_4peso.csv')
columna_a_predecir = 'pesada_4'

data, target = data_para_enrtenar(df, target=columna_a_predecir)

params = optimizo_modelo(data=data, target=target)

model, preds, rmse = entreno_modelo_final(params, data=data, target=target)

data['campa'] = data.index.str.split('_').str[1].str.strip().astype(int)
for col in data.columns:
    grafico_precision_modelo(target, preds, columna_a_predecir, rmse, valores_columna=data[col])

# %%


df = pd.read_csv('./dataset_3gan.csv')
columna_a_predecir = 'gancia_pesada_3'

data, target = data_para_enrtenar(df, target=columna_a_predecir)

params = optimizo_modelo(data=data, target=target)

model, preds, rmse = entreno_modelo_final(params, data=data, target=target)

grafico_precision_modelo(target, preds, columna_a_predecir, rmse)

# %%


df = pd.read_csv('./dataset_3peso.csv')
columna_a_predecir = 'pesada_3'

data, target = data_para_enrtenar(df, target=columna_a_predecir)

params = optimizo_modelo(data=data, target=target)

model, preds, rmse = entreno_modelo_final(params, data=data, target=target)

grafico_precision_modelo(target, preds, columna_a_predecir, rmse)