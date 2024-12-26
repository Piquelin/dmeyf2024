# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:12:56 2024

@author: jfgonzalez
"""
import polars as pl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import lightgbm as lgb

import optuna
from optuna.visualization.matplotlib import (plot_optimization_history,
                                             plot_param_importances,
                                             plot_slice, plot_contour)
import os
import gc
gc.collect()
# %% cargo parametros


dataset_path = './datasets/'
modelos_path = './modelos/'
db_path = './db/'
dataset_file = 'competencia_03.parquet'

ganancia_acierto = 273000
costo_estimulo = 7000

mes_train = [202106, 202105, 202102, 202101, 202104, 202103,
             202021, 202011, 202010, 202009, 202008, 202007,
             202006, 202105, 202002, 202001, 202004, 202003,
             201921, 201911, 201910, 201909, 201908, 201907,
            201906, 201905, 201904, 201903, 201902, 201901,
             ]
mes_test = 202109

mes_valid = 202107

# agregue sus semillas
semillas = [17, 19, 23, 29, 31]

# %%

dataset = pl.read_parquet('datasets/competencia_03.parquet')  # .filter(pl.col("foto_mes") > 202012)

dataset = dataset.drop(["tmobile_app", "cmobile_app_trx"])

# bajas = dataset.group_by(["foto_mes", "clase_ternaria"]).agg(pl.len().alias("conteo")).sort(["foto_mes", "clase_ternaria"]).to_pandas()

# %% drift


# Lista de meses y valores de UVA
vfoto_mes = [
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106,
    202107, 202108, 202109
]

vUVA = [
    2.001408838932958,  1.950325472789153,  1.89323032351521,
    1.8247220405493787, 1.746027787673673,  1.6871348409529485,
    1.6361678865622313, 1.5927529755859773, 1.5549162794128493,
    1.4949100586391746, 1.4197729500774545, 1.3678188186372326,
    1.3136508617223726, 1.2690535173062818, 1.2381595983200178,
    1.211656735577568,  1.1770808941405335, 1.1570338657445522,
    1.1388769475653255, 1.1156993751209352, 1.093638313080772,
    1.0657171590878205, 1.0362173587708712, 1.0,
    0.9669867858358365, 0.9323750098728378, 0.8958202912590305,
    0.8631993702994263, 0.8253893405524657, 0.7928918905364516,
    0.7666323845128089, 0.7428976357662823, 0.721615762047849
]

# Crear un DataFrame para los valores de UVA
uva_df = pl.DataFrame({
    "foto_mes": vfoto_mes,
    "UVA": vUVA
})

# Función para ajustar campos monetarios
def drift_uva(dataset: pl.DataFrame, campos_monetarios: list[str], uva_df: pl.DataFrame):
    dataset = dataset.join(uva_df, on="foto_mes", how="left")
    for campo in campos_monetarios:
        dataset = dataset.with_columns((pl.col(campo) * pl.col("UVA")).alias(campo))
    return dataset.drop("UVA")


# Filtrar columnas que correspondan a campos monetarios
campos_monetarios = [col for col in dataset.columns 
                     if col.startswith(("m", "Visa_m", "Master_m", "vm_m"))]

# dataset = drift_uva(dataset, campos_monetarios, uva_df)




# %% lag 1

cols_lagueables = dataset.columns[2:-1]

# Generar lags para cada columna en cols_lagueables, por grupo (numero_de_cliente)
lag = 1
lagged_columns = [
    (pl.col(col).shift(lag).over("numero_de_cliente").alias(f"{col}_lag{lag}"))
    for col in cols_lagueables
]
dataset = dataset.with_columns(lagged_columns)

# %% lag 2

lag = 2
lagged_columns = [
    (pl.col(col).shift(lag).over("numero_de_cliente").alias(f"{col}_lag{lag}"))
    for col in cols_lagueables
]
dataset = dataset.with_columns(lagged_columns)

# %% delta lag

# Calcular los delta lags
for col in cols_lagueables:
    dataset = dataset.with_columns([
        (pl.col(f'{col}') - pl.col(f"{col}_lag1")).alias(f"{col}_delta1"),
        (pl.col(f'{col}') - pl.col(f"{col}_lag2")).alias(f"{col}_delta2")
    ])


# %%

dataset.write_parquet("datasets/competencia_03_prepro.parquet")

# %%

dataset = pl.read_parquet('datasets/competencia_03_prepro.parquet')


# %% undersampleo


print(dataset['clase_ternaria'].value_counts())
print('Undersampleo')

# Agregar una columna 'azar' con valores aleatorios uniformes
dataset = dataset.with_columns(pl.lit(np.random.uniform(size=dataset.shape[0])).alias("azar"))

# Filtrar las filas según las condiciones
dataset = dataset.filter(
    ~(
        (pl.col("azar") > 0.02) &
        (pl.col("clase_ternaria") == "CONTINUA") &
        (pl.col("foto_mes").is_in(mes_train))
    )
)

# Eliminar la columna 'azar'
dataset = dataset.drop("azar")

print(dataset['clase_ternaria'].value_counts())


# %% Crear clase y pesos

dataset = dataset.with_columns(
    pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
    .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
    .otherwise(1.0)
    .alias("clase_peso")
)


# Crear columnas binarias
dataset = dataset.with_columns([
    (pl.col("clase_ternaria") == "BAJA+2").cast(int).alias("clase_binaria1"),
    (pl.col("clase_ternaria") != "CONTINUA").cast(int).alias("clase_binaria2"),
])
# %% Dividir en train y test
train_data = dataset.filter(pl.col("foto_mes").is_in(mes_train)).to_pandas()
test_data = dataset.filter(pl.col("foto_mes") == mes_test).to_pandas()

# valid_data =  dataset.filter(pl.col("foto_mes").is_in(mes_valid)).to_pandas()
# X_val
# y_val

# train_data = train_data[train_data['clase_ternaria']!='BAJA+1']

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


del dataset, dataset_path, dataset_file
# %% modelo


def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', round(((np.max(ganancia)/len(mes_train))/1000000),3) , True

def objective(trial):

    num_leaves = trial.suggest_int('num_leaves', 8, 1000),
    # learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3), # mas bajo, más iteraciones necesita
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 60, 8200),
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 0.6),
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 0.5),
    neg_bagging_fraction = trial.suggest_float('neg_bagging_fraction', 0.1, 1.0),

    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': num_leaves,
        'learning_rate': 0.03,  # learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'neg_bagging_fraction': neg_bagging_fraction,
        'seed': semillas[0],
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train,
                              label=y_train_binaria2, # eligir la clase
                              weight=w_train)
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=2000, # modificar, subit y subir... y descomentar la línea inferior
        callbacks=[lgb.early_stopping(stopping_rounds=100),],
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
study_name = 'exp_330_lgbm'

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

# %% optimizo

study.optimize(objective, n_trials=10)  # subir subir

# %%

plot_optimization_history(study)

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
    # 'learning_rate': study.best_trial.params['learning_rate'],
    'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
    'feature_fraction': study.best_trial.params['feature_fraction'],
    'bagging_fraction': study.best_trial.params['bagging_fraction'],
    'neg_bagging_fraction': study.best_trial.params['neg_bagging_fraction'],
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

model.save_model('modelos/lgb_27.0.5_sin1_lag1y2_del.txt')

model = lgb.Booster(model_file='./modelos/lgb_27.0.5_sin1_lag1y2_del.txt')

# %%

y_pred_lgm = model.predict(X_test)

# %%

y_pred = pd.Series(y_pred_lgm, index=X_test.numero_de_cliente)

y_pred = X_test[['numero_de_cliente', 'foto_mes']]
y_pred['clase_ternaria'] = y_test_class
y_pred['pred'] = y_pred_lgm

# %%

def armo_entregas_desde_probs(df_, modelos=1, semillas=1):
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
        for corte in range(8000, 13001, 1000):
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
lista = armo_entregas_desde_probs(y_pred)

# %%
import os

guardo_en_archivos(lista, experimento='c03_local_31.0.03_LD1y2_p')




# %% uno modelos


file = 'SC/expw_SC-0035_tb_future_prediccion.txt'
df_ = pd.read_csv(file, sep='\t')
lista = lista + armo_entregas_desde_probs(df_, modelos=1, semillas=15)


lista[0].index = lista[0].numero_de_cliente
lista[1].index = lista[1].numero_de_cliente
nuevo = pd.concat([lista[0].prom, lista[1].prom], axis=1)

l_nuevo = armo_entregas_desde_probs(df_, modelos=1, semillas=2)


guardo_en_archivos(l_nuevo, experimento='c03_local_27_LD1y2_y_SC0035')

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
