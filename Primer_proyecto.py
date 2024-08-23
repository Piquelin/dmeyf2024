# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:58:02 2024

@author: jfgonzalez
"""


import polars as pl

dataset = pl.read_csv("competencia_01_crudo.csv", infer_schema_length=10000)


#%% pivoteo

# pivoteo con las fechas para compara los meses. tomo el valor de active _quarter para simplificar los reemplazos
tabla = dataset[['numero_de_cliente', 'foto_mes', 'active_quarter']].pivot(columns='foto_mes', index='numero_de_cliente', values='active_quarter')


# Reemplazar valores nulos por False y valores enteros por True
tabla_bool = tabla.with_columns([
    pl.col(column).is_not_null().alias(column)
    for column in tabla.columns[1:]
])


#%% funcion de evaluación

# con esto evalúo cada uno de los primeros 4 meses en función a los dos siguientes
# para el caso de clientes intermitentes se los considera como que se fueron en el mes en cuestión
# y se evalúa nuevamente a partir que se reincorporan.

def evaluar_condiciones(tabla: pl.DataFrame, col1: str, col2: str, col3: str) -> pl.Expr:

    evaluacion = (
        pl.when(pl.col(col1) == False)
        .then(None)
        .when(
            (pl.col(col1) == True) & (pl.col(col2) == True) & (pl.col(col3) == True)
        )
        .then(pl.lit("CONTINUA"))
        .when(
            (pl.col(col1) == True) & (pl.col(col2) == True) & (pl.col(col3) == False)
        )
        .then(pl.lit("BAJA+2"))
        .when(
            (pl.col(col1) == True) & (pl.col(col2) == False) 
        )
        .then(pl.lit("BAJA+1"))
        .otherwise(None)
    )

    return evaluacion

#%% Genero los resultados para los 4 primeros meses

lista_meses = [	'202101','202102','202103','202104','202105','202106']

for i in range(4):
    tabla_bool = tabla_bool.with_columns(evaluar_condiciones(tabla_bool, lista_meses[i], lista_meses[i+1], lista_meses[i+2]).alias(f"evaluacion_{lista_meses[i]}"))


#%% unpivoteo y hago join

clase_cliente = tabla_bool.unpivot(index='numero_de_cliente')
clase_cliente = clase_cliente.rename({
    "variable": "foto_mes",
    'value':'clase_ternaria'})
# descarto columas auxiliares
clase_cliente = clase_cliente.filter(pl.col("foto_mes").str.starts_with("e"))
# paso en limpio las fechas
clase_cliente = clase_cliente.with_columns(pl.col("foto_mes").str.slice(-6).cast(pl.Int64).alias("foto_mes"))

#%%  Realizar la unión usando 'numero_de' y 'foto_mes' como claves
resultado = dataset.join(
    clase_cliente,
    on=["numero_de_cliente", "foto_mes"],
    how="left"  
)

# guardo
resultado.write_csv("resultado_con_clase.csv", separator=",")
resultado.write_parquet("resultado_con_clase.parquet")

#%%

import polars as pl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree,  _tree


resultado = pl.read_parquet('resultado_con_clase.parquet')
tabla_202104 = resultado.filter(pl.col("foto_mes") == 202104)


#%%

X = tabla_202104.to_pandas().drop('clase_ternaria', axis=1)
y = tabla_202104.to_pandas()['clase_ternaria']
y.value_counts(normalize=True)*100


resultado_filtrado = tabla_202104.filter(pl.col("clase_ternaria").is_null())


#%%


model = DecisionTreeClassifier(criterion='gini',
                               random_state=17,
                               min_samples_split=80,
                               min_samples_leaf=160,
                               max_depth=10)

model.fit(X, y)

features = pd.DataFrame([model.feature_names_in_, model.feature_importances_]).T
    
#%%

from imblearn.over_sampling import SMOTE, ADASYN

X_resampled, y_resampled = SMOTE().fit_resample(X.fillna(0), y)


model.fit(X_resampled, y_resampled)

features_r = pd.DataFrame([model.feature_names_in_, model.feature_importances_]).T



# %% Visualización

# Todo es mucho más bonito si tiene colores

plt.figure(figsize=(40,20))
plot_tree(model, feature_names=X.columns, filled=True, class_names=model.classes_, rounded=True, impurity=False) #,  proportion=True)
plt.show()
     
#%%

def get_leaf_info(tree):
    tree_ = tree.tree_
    class_names = model.classes_
    leaf_info = []
    for i in range(tree_.node_count):
        if tree_.children_left[i] == _tree.TREE_LEAF:
            class_counts = tree_.value[i][0]
            predicted_class_index = class_counts.argmax()
            predicted_class = class_names[predicted_class_index]
            row = {
                'Node': i,
                'Samples': int(tree_.n_node_samples[i]),
                'Predicted Class': predicted_class
            }
            for j, class_name in enumerate(class_names):
                row[class_name] = int(class_counts[j])
            leaf_info.append(row)
    return pd.DataFrame(leaf_info)

leaf_df = get_leaf_info(model)
leaf_df

#%%


ganancia_acierto = 273000
costo_estimulo = 7000

leaf_df["ganancia"] = ganancia_acierto*leaf_df["BAJA+2"] - costo_estimulo*leaf_df["BAJA+2"] + -costo_estimulo*leaf_df["CONTINUA"]
leaf_df.sort_values("ganancia", ascending=False, inplace=True)
leaf_df
     

# Vaya! no son pocos las hojas que hubieran dado ganancia positiva, y todas a su vez estuvieron mal clasificadas por el modelo.

# Veamos de calcular la probabilidad de BAJA+2 por cada hoja

leaf_df["prob_baja_2"] = leaf_df["BAJA+2"]/leaf_df["Samples"]
leaf_df.sort_values("prob_baja_2", ascending=False, inplace=True)
leaf_df
     

# Finalmente calculemos la ganancia acumulada para cada posible punto de corte:

leaf_df['gan_acumulada'] = leaf_df['ganancia'].cumsum()
leaf_df
     
#%%

