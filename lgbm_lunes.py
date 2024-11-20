# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:12:56 2024

@author: jfgonzalez
"""

# %%
#%pip install scikit-learn==1.3.2
#%pip install seaborn==0.13.1
#%pip install numpy==1.26.4
#%pip install matplotlib==3.7.1
#%pip install pandas==2.1.4
#%pip install lightgbm==4.4.0
#%pip install optuna==3.6.1

# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
# from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import lightgbm as lgb

import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice, plot_contour

from time import time

import pickle

#%%


base_path = 'C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/'
dataset_path = base_path + 'datasets/'
modelos_path = base_path + 'modelos/'
db_path = base_path + 'db/'
dataset_file = 'competencia_02.parquet'

ganancia_acierto = 273000
costo_estimulo = 7000

mes_train = 202104
mes_test = 202106

# agregue sus semillas
semillas = [17,19,23,29,31]

data = pd.read_parquet(dataset_path + dataset_file)

# %%


data['clase_peso'] = 1.0

data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

data['clase_binaria1'] = 0
data['clase_binaria2'] = 0
data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)

data.tmobile_app = data.tmobile_app.fillna(np.nan).astype('float64')
data.cmobile_app_trx = data.cmobile_app_trx.fillna(np.nan).astype('float64')

# %%


train_data = data[data['foto_mes'] == mes_train]
test_data = data[data['foto_mes'] == mes_test]

X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_train_binaria1 = train_data['clase_binaria1']
y_train_binaria2 = train_data['clase_binaria2']
w_train = train_data['clase_peso']

X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_test_binaria1 = test_data['clase_binaria1']
y_test_class = test_data['clase_ternaria']
w_test = test_data['clase_peso']

# %%

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
Xif = imp_mean.fit_transform(X_test)

# %%

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True

# Parámetros del modelos.
params = {
    'objective': 'binary',
    'metric': 'gan_eval',
    'boosting_type': 'gbdt',
    'max_bin': 31,
    'num_leaves': 150,
    'learning_rate': 0.02,
    'feature_fraction': 0.3,
    'bagging_fraction': 0.7,
    'verbose': 0
}


# %%

train_data1 = lgb.Dataset(X_train, label=y_train_binaria1, weight=w_train)
train_data2 = lgb.Dataset(X_train, label=y_train_binaria2, weight=w_train)

# %%

cv_results1 = lgb.cv(
    params,
    train_data1,
    num_boost_round=150,
    feval=lgb_gan_eval,
    nfold=5,
    seed=semillas[0]
)

cv_results2 = lgb.cv(
    params,
    train_data2,
    num_boost_round=150,
    feval=lgb_gan_eval,
    nfold=5,
    seed=semillas[0]
)

# %%

df_ganancias = pd.DataFrame({
    'binaria1': cv_results1['valid gan_eval-mean'],
    'binaria2': cv_results2['valid gan_eval-mean'],
    'Iteracion': range(1, len(cv_results1['valid gan_eval-mean']) + 1)
})

# Normalizamos la ganancias
df_ganancias['binaria1'] = df_ganancias['binaria1']*5
df_ganancias['binaria2'] = df_ganancias['binaria2']*5

plt.figure(figsize=(10, 6))
sns.lineplot(x='Iteracion', y='binaria1', data=df_ganancias, label='binaria 1')
sns.lineplot(x='Iteracion', y='binaria2', data=df_ganancias, label='binaria 2')
plt.title('Comparación de las Ganancias de las 2 clases binarias')
plt.xlabel('Iteración')
plt.ylabel('Ganancia')
plt.legend()
plt.show()


# %%

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


storage_name = "sqlite:///" + db_path + "optimization_lgbm.db"
study_name = 'exp_302_lgbm'

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

# %%

study.optimize(objective, n_trials=20) # subir subir

# %%

plot_optimization_history(study)


# %%

plot_param_importances(study)


# %%

plot_slice(study)


plot_contour(study)


plot_contour(study, params=['num_leaves','min_data_in_leaf'] )

plot_contour(study, params=['bagging_fraction','feature_fraction'] )

plot_contour(study, params=['learning_rate','feature_fraction'] )
plot_contour(study, params=['learning_rate','min_data_in_leaf'] )
plot_contour(study, params=['learning_rate','bagging_fraction'] )
plot_contour(study, params=['learning_rate','num_leaves'] )




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
    'verbose': 0
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
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df[importance_df['importance'] > 0]

# %%

model.save_model('./modelos/lgb_first.txt')

model = lgb.Booster(model_file='./modelos/lgb_first.txt')

# %%

y_pred_lgm = model.predict(X_test)

# %%

def ganancia_prob(y_pred, y_true, prop = 1):
  ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)
  return ganancia[y_pred >= 0.025].sum() / prop

print("Ganancia LGBM:", ganancia_prob(y_pred_lgm, y_test_binaria1))

# %%

ganancia = np.where(y_test_binaria1 == 1, ganancia_acierto, 0) - np.where(y_test_binaria1 == 0, costo_estimulo, 0)

idx = np.argsort(y_pred_lgm)[::-1]

ganancia = ganancia[idx]
y_pred_lgm = y_pred_lgm[idx]

ganancia_cum = np.cumsum(ganancia)

# %%

piso_envios = 4000
techo_envios = 20000

plt.figure(figsize=(10, 6))
plt.plot(y_pred_lgm[piso_envios:techo_envios], ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
plt.title('Curva de Ganancia')
plt.xlabel('Predicción de probabilidad')
plt.ylabel('Ganancia')
plt.axvline(x=0.025, color='g', linestyle='--', label='Punto de corte a 0.025')
plt.legend()
plt.show()

# %%

piso_envios = 4000
techo_envios = 20000

ganancia_max = ganancia_cum.max()
gan_max_idx = np.where(ganancia_cum == ganancia_max)[0][0]

plt.figure(figsize=(10, 6))
plt.plot(range(piso_envios, len(ganancia_cum[piso_envios:techo_envios]) + piso_envios), ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
plt.axvline(x=gan_max_idx, color='g', linestyle='--', label=f'Punto de corte a la ganancia máxima {gan_max_idx}')
plt.axhline(y=ganancia_max, color='r', linestyle='--', label=f'Ganancia máxima {ganancia_max}')
plt.title('Curva de Ganancia')
plt.xlabel('Clientes')
plt.ylabel('Ganancia')
plt.legend()
plt.show()

# %%

def analisis_1(semilla):
  df_cut_point = pd.DataFrame({'ganancia': ganancia, 'y_pred_lgm': y_pred_lgm})

  private_idx, public_idx = train_test_split(df_cut_point.index, test_size=0.3, random_state=semilla, stratify=y_test_binaria1)

  df_cut_point['public'] = 0.0
  df_cut_point['private'] = 0.0
  df_cut_point.loc[private_idx, 'private'] = ganancia[private_idx] / 0.7
  df_cut_point.loc[public_idx, 'public'] = ganancia[public_idx] / 0.3

  df_cut_point['nro_envios'] = df_cut_point.reset_index().index

  df_cut_point['public_cum'] = df_cut_point['public'].cumsum()
  df_cut_point['private_cum'] = df_cut_point['private'].cumsum()

  plt.figure(figsize=(10, 6))
  plt.plot(df_cut_point['nro_envios'][4000:20000], df_cut_point['public_cum'][4000:20000], label='Ganancia Pública Acumulada')
  plt.plot(df_cut_point['nro_envios'][4000:20000], df_cut_point['private_cum'][4000:20000], label='Ganancia Privada Acumulada')

  max_public_cum = df_cut_point['public_cum'][4000:20000].max()
  max_public_idx = df_cut_point['public_cum'][4000:20000].idxmax()
  plt.axvline(x=max_public_idx, color='g', linestyle='--', label=f'Máximo Ganancia Pública en {max_public_idx}')

  max_private_cum = df_cut_point['private_cum'][4000:20000].max()
  max_private_idx = df_cut_point['private_cum'][4000:20000].idxmax()
  plt.axvline(x=max_private_idx, color='r', linestyle='--', label=f'Máximo Ganancia Privada en {max_private_idx}')

  plt.title('Curva de Ganancia Pública y Privada')
  plt.xlabel('Número de envíos')
  plt.ylabel('Ganancia Acumulada')
  plt.legend()
  plt.show()

analisis_1(semillas[1])

# %%
def analisis_2(semilla, desde, paso, cantidad, private = False):

  df_cut_point = pd.DataFrame({'ganancia': ganancia, 'y_pred_lgm': y_pred_lgm})
  df_cut_point['nro_envios'] = df_cut_point.reset_index().index

  plt.figure(figsize=(10, 6))

  private_idx, public_idx = train_test_split(df_cut_point.index, test_size=0.3, random_state=semilla, stratify=y_test_binaria1)

  df_cut_point['public'] = 0.0
  df_cut_point.loc[public_idx, 'public'] = ganancia[public_idx] / 0.3
  df_cut_point['public_cum'] = df_cut_point['public'].cumsum()

  maximo_paso = desde + paso*cantidad
  plt.plot(df_cut_point['nro_envios'][list(range(desde, maximo_paso + 1, paso))], df_cut_point['public_cum'][list(range(desde, maximo_paso + 1, paso))], label='Ganancia Pública Acumulada')
  max_public_cum = df_cut_point['public_cum'][list(range(desde, maximo_paso + 1, paso))].max()
  max_public_idx = df_cut_point['public_cum'][list(range(desde, maximo_paso + 1, paso))].idxmax()
  plt.axvline(x=max_public_idx, color='g', linestyle='--', label=f'Máximo Ganancia Pública en {max_public_idx}')

  if private:
    df_cut_point['private'] = 0.0
    df_cut_point.loc[private_idx, 'private'] = ganancia[private_idx] / 0.7
    df_cut_point['private_cum'] = df_cut_point['private'].cumsum()
    plt.plot(df_cut_point['nro_envios'][4000:20000], df_cut_point['private_cum'][4000:20000], label='Ganancia Privada Acumulada')
    max_private_cum = df_cut_point['private_cum'][4000:20000].max()
    max_private_idx = df_cut_point['private_cum'][4000:20000].idxmax()
    plt.axvline(x=max_private_idx, color='r', linestyle='--', label=f'Máximo Ganancia Privada en {max_private_idx}')

  plt.title('Curva de Ganancia Pública y Privada')
  plt.xlabel('Número de envíos')
  plt.ylabel('Ganancia Acumulada')
  plt.legend()
  plt.show()

analisis_2(semillas[0], 4000, 500, 25, private=True)

# %%