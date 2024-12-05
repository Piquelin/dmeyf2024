# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:05:59 2024

@author: Piquelin
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import gc

import optuna
from optuna.trial import Trial
from datetime import datetime


dataset = pl.read_parquet('datasets/competencia_03.parquet').filter(pl.col("foto_mes") > 202012)

dataset = dataset.drop(["tmobile_app", "cmobile_app_trx"])

# bajas = dataset.group_by(["foto_mes", "clase_ternaria"]).agg(pl.len().alias("conteo")).sort(["foto_mes", "clase_ternaria"]).to_pandas()

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
        # (pl.col(f'{col}') - pl.col(f"{col}_lag2")).alias(f"{col}_delta2")
    ])



# %%

# Definir los campos buenos (excluir 'clase_ternaria')
campos_buenos = [col for col in dataset.columns if col != "clase_ternaria"]

# Semilla para reproducibilidad
np.random.seed(799991)

# Agregar columna 'azar' con números aleatorios uniformes
dataset = dataset.with_columns(pl.lit(np.random.uniform(size=dataset.shape[0])).alias("azar"))

# Filtrar datos futuros
dfuture = dataset.filter(pl.col("foto_mes") == 202109)

# Undersampling de "CONTINUA" al 2%
dataset = dataset.with_columns(
    ((pl.col("foto_mes") <= 202107) &
     (
         (pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"])) |
         (pl.col("azar") < 0.02)
     )).alias("fold_train")
)

# Crear la columna 'clase01'
dataset = dataset.with_columns(
    pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0).otherwise(1).alias("clase01")
)

# %% Preparar los datos para LightGBM
# Validación
dvalidate = lgb.Dataset(
    data=dataset.filter(pl.col("foto_mes") == 202107).select(campos_buenos).to_numpy(),
    label=dataset.filter(pl.col("foto_mes") == 202107)["clase01"].to_numpy(),
    free_raw_data=True
)

# %%
# Entrenamiento con pesos ajustados
train_filter = dataset["fold_train"]
dtrain = lgb.Dataset(
    data=dataset.filter(train_filter).select(campos_buenos).to_numpy(),
    label=dataset.filter(train_filter)["clase01"].to_numpy(),
    weight=dataset.filter(train_filter)
        .select(pl.when(pl.col("foto_mes") <= 202106).then(1.0).otherwise(0.0))
        .to_numpy(),
    free_raw_data=True
)



# %%


def objective(trial):

    num_leaves = trial.suggest_int('num_leaves', 8, 1000),
    # learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3), # mas bajo, más iteraciones necesita
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 500, 1000),
    # feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0),
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0),

    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        "force_row_wise": True,  # para evitar warning
        'max_bin': 31,
        'num_leaves': num_leaves,
        # 'learning_rate': learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        # 'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'seed': 799991,
        "learning_rate": 0.03,
        "feature_fraction": 0.5,
        # 'verbose': -1


    }
    # train_data = lgb.Dataset(dtrain,
    #                           label=y_train_binaria2, # eligir la clase
    #                           weight=w_train)
    cv_results = lgb.cv(

        params,
        train_set=dtrain,
        valid_sets=[dvalidate],
        valid_names=["valid"],

        # params,
        # train_data,
        # num_boost_round=1000, # modificar, subit y subir... y descomentar la línea inferior
        # callbacks=[lgb.early_stopping(stopping_rounds=50),],
        # # early_stopping_rounds= 50, # int(50 + 5 / learning_rate),
        # feval=lgb_gan_eval,
        # stratified=True,
        # nfold=5,
        # seed=semillas[0]
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


# Definir la función de optimización de AUC
def EstimarGanancia_AUC_lightgbm(trial: Trial):
    # print(datetime.now().strftime("%a %b %d %X %Y"))

    # Parámetros sugeridos
    num_leaves = trial.suggest_int("num_leaves", 8, 1024)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 64, 8192)

    # Parámetros específicos de entrenamiento
    param_train = {
        "num_iterations": 2048,
        "early_stopping_rounds": 200,
    }

    # Combinar parámetros básicos
    param_completo = {**param_basicos, **param_train, "num_leaves": num_leaves, "min_data_in_leaf": min_data_in_leaf}

    # Entrenar el modelo
    modelo_train = lgb.train(
        params=param_completo,
        train_set=dtrain,
        valid_sets=[dvalidate],
        valid_names=["valid"],
        # verbose_eval=-100
    )

    # Calcular el AUC
    AUC = modelo_train.best_score["valid"]["auc"]

    # Mensajes de depuración
    print(f"AUC obtenido: {AUC}")
    print(f"Parámetros: num_leaves={num_leaves}, min_data_in_leaf={min_data_in_leaf}")

    # Liberar memoria
    del modelo_train
    gc.collect()

    return AUC,


# Crear un estudio de Optuna
study = optuna.create_study(direction="maximize")

# Ejecutar la optimización de AUC con 10 iteraciones
study.optimize(EstimarGanancia_AUC_lightgbm, n_trials=3)


# Obtener el mejor AUC y los mejores parámetros
best_AUC = study.best_value
best_params = study.best_params

print(f"Mejor AUC: {best_AUC}")
print(f"Mejores parámetros: {best_params}")


# %%


# Mostrar un subconjunto del dataset para verificar
print(dataset.columns)
