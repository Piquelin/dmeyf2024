# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:12:56 2024

@author: jfgonzalez
"""

# %%
# %pip install scikit-learn==1.3.2
# %pip install seaborn==0.13.1
# %pip install numpy==1.26.4
# %pip install matplotlib==3.7.1
# %pip install pandas==2.1.4
# %pip install lightgbm==4.4.0
# %pip install optuna==3.6.1

# %%

import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer

import lightgbm as lgb
from tqdm import tqdm
import optuna
from optuna.visualization.matplotlib import (plot_optimization_history,
                                             plot_param_importances,
                                             plot_slice, plot_contour)
import os
import gc
gc.collect()
# %% cargo


base_path = ('E:/Users/Piquelin/Documents/Maestría_DataMining/' +
             'Economia_y_finanzas/')
dataset_path = base_path + 'datasets/'
modelos_path = base_path + 'modelos/'
db_path = base_path + 'db/'
dataset_file = 'competencia_03.parquet'

ganancia_acierto = 273000
costo_estimulo = 7000

mes_train = [202107, 202106, 202105, 202102, 202101, #  202104, 202103,
             202021, 202011, 202010, 202009, 202008, 202007,
             202006, 202105, 202002, 202001, #  202004, 202003,
             201921, 201911, 201910, 201909, 201908, 201907,
            # 201906, 201905, 201904, 201903, 201902, 201901,
             ]
mes_test = 202109

# agregue sus semillas
semillas = [17, 19, 23, 29, 31]

data = pd.read_parquet(dataset_path + dataset_file)
data = data.drop(columns=['tmobile_app', 'cmobile_app_trx'])

# %% lag 1


cols_lagueables = data.columns[2:-1]

lag=1

# Generar las columnas con lags
for col in tqdm(cols_lagueables):
    data[f"{col}_lag{lag}"] = (
        data.groupby("numero_de_cliente")[col].shift(lag))
lag=2

# Generar las columnas con lags
for col in tqdm(cols_lagueables):
    data[f"{col}_lag{lag}"] = (
        data.groupby("numero_de_cliente")[col].shift(lag))

# %% undersampleo

data["azar"] = np.random.uniform(size=len(data))

data = data[~((data["azar"] > 0.02)
              & (data["clase_ternaria"] == "CONTINUA")
              & (data["foto_mes"].isin(mes_train)))]

data = data.drop(columns=['azar'])

print(data.clase_ternaria.value_counts())

# %% divido train y test

train_data = data[data['foto_mes'].isin(mes_train)]

train_data = data[data['foto_mes'].isin(mes_train)]
test_data = data[data['foto_mes'] == mes_test]

X_train = train_data.drop(['clase_ternaria', 'clase_peso',
                           'clase_binaria1', 'clase_binaria2'], axis=1)
y_train_binaria1 = train_data['clase_binaria1']
y_train_binaria2 = train_data['clase_binaria2']
w_train = train_data['clase_peso']

X_test = test_data.drop(['clase_ternaria', 'clase_peso',
                         'clase_binaria1', 'clase_binaria2'], axis=1)
y_test_binaria1 = test_data['clase_binaria1']
y_test_class = test_data['clase_ternaria']
w_test = test_data['clase_peso']


# %% creo clase y pesos

data['clase_peso'] = 1.0

data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

data['clase_binaria1'] = 0
data['clase_binaria2'] = 0
data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)


print(data.clase_ternaria.value_counts())
# %% modelo


def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia)/len(mes_train) , True

def objective(trial):

    num_leaves = trial.suggest_int('num_leaves', 8, 1000),
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3), # mas bajo, más iteraciones necesita
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 500, 1000),
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0),
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0),

    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'seed': semillas[0],
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train,
                              label=y_train_binaria2, # eligir la clase
                              weight=w_train)
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000, # modificar, subit y subir... y descomentar la línea inferior
        callbacks=[lgb.early_stopping(stopping_rounds=50),],
        # early_stopping_rounds= 50, # int(50 + 5 / learning_rate),
        feval=lgb_gan_eval,
        stratified=True,
        nfold=5,
        seed=semillas[0]
    )
    max_gan = max(cv_results['valid gan_eval-mean'])
    best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1

    # Guardamos cual es la mejor iteración del modelo
    trial.set_user_attr("best_iter", best_iter)

    return max_gan * 5


# %% creo estudio

storage_name = "sqlite:///" + db_path + "optimization_lgbm.db"
study_name = 'exp_314_lgbm'

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

# %% optimizo

study.optimize(objective, n_trials=1)  # subir subir

# %%

plot_optimization_history(study)


# %%

plot_param_importances(study)


# %%

plot_slice(study)


plot_contour(study)


plot_contour(study, params=['num_leaves', 'min_data_in_leaf'])

plot_contour(study, params=['bagging_fraction', 'feature_fraction'])

plot_contour(study, params=['learning_rate', 'feature_fraction'])
plot_contour(study, params=['learning_rate', 'min_data_in_leaf'])
plot_contour(study, params=['learning_rate', 'bagging_fraction'])
plot_contour(study, params=['learning_rate', 'num_leaves'])




df = study.trials_dataframe()

# %%

best_iter = study.best_trial.user_attrs["best_iter"]
print(f"Mejor cantidad de árboles para el mejor model {best_iter}")
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'max_bin': 31,
    'num_leaves': study.best_trial.params['num_leaves'],
    'learning_rate': study.best_trial.params['learning_rate'],
    'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
    'feature_fraction': study.best_trial.params['feature_fraction'],
    'bagging_fraction': study.best_trial.params['bagging_fraction'],
    'seed': semillas[0],
    'verbose': -1
}

train_data = lgb.Dataset(X_train,
                         label=y_train_binaria2,
                         weight=w_train)

model = lgb.train(params,
                  train_data,
                  num_boost_round=best_iter)

# %%

lgb.plot_importance(model, figsize=(10, 20))
plt.show()

# %%

importances = model.feature_importance()
feature_names = X_train.columns.tolist()
importance_df = pd.DataFrame({'feature': feature_names,
                              'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df[importance_df['importance'] > 0]

# %%

model.save_model('modelos/lgb_27_lag1y2.txt')

model = lgb.Booster(model_file='./modelos/lgb_27_lag1y2.txt')

# %%

y_pred_lgm = model.predict(X_test)

# %%

y_pred = pd.Series(y_pred_lgm, index=X_test.numero_de_cliente)

y_pred = X_test[['numero_de_cliente', 'foto_mes']]
y_pred['clase_ternaria'] = y_test_class
y_pred['pred'] = y_pred_lgm

# %%

def armo_entregas_desde_probs(df_, modelos=3, semillas=20):
    lista_prob_prom = []

    for i in range(modelos):
        df_entrega = df_[['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
        # df_entrega['numero_de_cliente'] = df_['numero_de_cliente']
        df_entrega['prom'] = df_.iloc[:, (i*semillas + 3):(i*semillas + 3 + semillas)].T.mean()
        df_entrega = df_entrega.sort_values('prom', ascending=False).reset_index()
        df_entrega = df_entrega[['numero_de_cliente',  'foto_mes', 'clase_ternaria', 'prom']]
        lista_prob_prom.append(df_entrega)

    total_clientes = len(lista_prob_prom[0])
    for i in range(modelos):
        for corte in range(8000, 16001, 500):
            array = np.zeros((total_clientes, 1))
            array[:corte] = 1
            lista_prob_prom[i][f'pred_{corte}'] = array.astype(int)

    return lista_prob_prom


def guardo_en_archivos(dfs, experimento):
    # Crear directorio de entregas
    dir_entregas = f'entregas_{experimento}'
    os.makedirs(dir_entregas, exist_ok=True)

    # Guardar predicciones de cada modelo en archivos CSV
    for modelo, df in enumerate(dfs):
        for col in df.columns:
            if col.startswith('pred_'):
                corte = col.split('_')[1]  # Extraer el número del corte
                df_pred = df[['numero_de_cliente', col]].copy()
                df_pred.columns = ['numero_de_cliente', 'Predicted']

                # Nombre de archivo y ruta de guardado
                archivo_nombre = f"{experimento}_{modelo}_{corte}.csv"
                archivo_ruta = os.path.join(dir_entregas, archivo_nombre)

                # Guardar en CSV
                df_pred.to_csv(archivo_ruta, index=False)
    return None

# %%


# df_ = df_[df_['foto_mes'] == 202109]
lista = armo_entregas_desde_probs(y_pred, modelos=1, semillas=1)

# %%
import os

guardo_en_archivos(l_nuevo, experimento='c03_local_L1y2')




# %% uno modelos


file = 'SC/expw_SC-0035_tb_future_prediccion.txt'
df_ = pd.read_csv(file, sep='\t')
lista = lista + armo_entregas_desde_probs(df_, modelos=1, semillas=15)


lista[0].index = lista[0].numero_de_cliente
lista[1].index = lista[1].numero_de_cliente
nuevo = pd.concat([lista[0].prom, lista[1].prom], axis=1)

l_nuevo = armo_entregas_desde_probs(df_, modelos=1, semillas=2)




# %%

ganancia = (np.where(y_test_binaria1 == 1, ganancia_acierto, 0) -
            np.where(y_test_binaria1 == 0, costo_estimulo, 0))

idx = np.argsort(y_pred_lgm)[::-1]

ganancia = ganancia[idx]
y_pred_lgm = y_pred_lgm[idx]

ganancia_cum = np.cumsum(ganancia)

# %%

piso_envios = 0
techo_envios = 30000

plt.figure(figsize=(10, 6))
plt.plot(y_pred_lgm[piso_envios:techo_envios],
         ganancia_cum[piso_envios:techo_envios],
         label='Ganancia LGBM')
plt.title('Curva de Ganancia')
plt.xlabel('Predicción de probabilidad')
plt.ylabel('Ganancia')
plt.axvline(x=0.025, color='g', linestyle='--', label='Punto de corte a 0.025')
plt.legend()
plt.show()
