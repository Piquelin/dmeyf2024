# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 08:36:33 2024

@author: jfgonzalez
"""


#%%

import polars as pl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree,  _tree


resultado = pl.read_parquet('resultado_con_clase.parquet')
tabla_202104 = resultado.filter(pl.col("foto_mes") <= 202104)

# Suponiendo que tienes un DataFrame llamado df
# Primero define los sufijos
suffixes = [
    '_delinquency', '_status', '_mfinanciacion_limite', '_Fvencimiento',
    '_Finiciomora', '_msaldototal', '_msaldopesos', '_msaldodolares',
    '_mconsumospesos', '_mconsumosdolares', '_mlimitecompra',
    '_madelantopesos', '_madelantodolares', '_fultimo_cierre',
    '_mpagado', '_mpagospesos', '_mpagosdolares', '_fechaalta',
    '_mconsumototal', '_cconsumos', '_cadelantosefectivo', '_mpagominimo'
]

# Luego sumamos las columnas correspondientes por cada sufijo
for suffix in suffixes:
    tabla_202104 = tabla_202104.with_columns(
        (pl.col(f'Master{suffix}') + pl.col(f'Visa{suffix}')).alias(f'Total{suffix}')
    )

# Ahora df tiene columnas sumadas por cada sufijo

tabla_202104
#%%

X = tabla_202104.to_pandas().drop('clase_ternaria', axis=1)
y = tabla_202104.to_pandas()['clase_ternaria']
print(y.value_counts(normalize=True)*100)


# resultado_filtrado = tabla_202104.filter(pl.col("clase_ternaria").is_null())


#%%


model = DecisionTreeClassifier(criterion='gini',
                               random_state=17,
                               min_samples_split=250,
                               min_samples_leaf=100,
                               max_leaf_nodes=16,
                               max_depth=7,
                               class_weight=None,
                               min_impurity_decrease=0.000)

model.fit(X, y)

features = pd.DataFrame([model.feature_names_in_, model.feature_importances_]).T


# %% Visualización

# Todo es mucho más bonito si tiene colores

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, filled=True,
          class_names=model.classes_, rounded=True,
          impurity=True, fontsize=8,  proportion=False,
          node_ids=True, )
plt.show()
     

#%%


def calcular_ganancia_polars(model):
    # Obteniendo el árbol y los valores asociados
    tree = model.tree_
    nodos = tree.node_count
    valores = tree.value
    pesos = tree.weighted_n_node_samples

    # Creando una lista para almacenar los datos de cada nodo
    data = []

    for i in range(nodos):
        # Para cada nodo, obtenemos la predicción (clase mayoritaria) y los valores de cada clase multiplicados por los pesos
        prediccion = np.argmax(valores[i])
        clase_0 = int(valores[i][0][0] * pesos[i])  # Cantidad de la clase 0 en el nodo multiplicada por el peso
        clase_1 = int(valores[i][0][1] * pesos[i])  # Cantidad de la clase 1 en el nodo multiplicada por el peso
        clase_2 = int(valores[i][0][2] * pesos[i])  # Cantidad de la clase 2 en el nodo multiplicada por el peso

        data.append([i, prediccion, clase_0, clase_1, clase_2])

    # Creando el DataFrame en Polars
    df_nodos = pl.DataFrame(
        data,orient="row",
        schema=['Nodo', 'Predicción', 'BAJA+1', 'BAJA+2', 'CONTINUA']
    )

    # Calcular las ganancias
    df_nodos = df_nodos.with_columns([
        (pl.col('BAJA+2') * 273000).alias('ganancia1'),
        (-pl.col('BAJA+1') * 7000).alias('ganancia2'),
        (-pl.col('CONTINUA') * 7000).alias('ganancia3'),
        (pl.col('BAJA+1') + pl.col('CONTINUA')).alias('Otros'),
        (pl.col('BAJA+2') * 273000 - pl.col('BAJA+1') * 7000 - pl.col('CONTINUA') * 7000).alias('ganancia'),
        ((pl.col('BAJA+2') / (pl.col('BAJA+1') + pl.col('BAJA+2') + pl.col('CONTINUA'))) * 100).alias('prob_+2')
    ])

    # Extraer información sobre los nodos
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right

    # Identificar hojas
    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_indices = [i for i in range(n_nodes) if is_leaf[i]]

    # Mostrar el DataFrame para los nodos hoja
    df_hojas = df_nodos.filter(pl.col('Nodo').is_in(leaf_indices))
    
    df_resultado = df_hojas.select(['Otros', 'BAJA+2', 'ganancia', 'prob_+2']).sort('prob_+2', descending=True)
    
    with pl.Config(
    tbl_cell_numeric_alignment="RIGHT",
    thousands_separator=True,
    float_precision=2,
    tbl_rows=20
    ):
        pl.Config.set_tbl_cell_alignment("RIGHT")
        print(df_resultado)

    return df_resultado

df_resultado = calcular_ganancia_polars(model)

#%%


tabla_202106 = resultado.filter(pl.col("foto_mes") == 202106)

tabla_202106 = tabla_202106.to_pandas()

pred = tabla_202106['ctrx_quarter'] <= 3.5
tabla_202106['Predicted'] = pred.astype(int)

tabla_202106[['numero_de_cliente', 'Predicted']].to_csv('pred.csv', index=False)
tabla_202106['Predicted'] = 1
tabla_202106[['numero_de_cliente', 'Predicted']].to_csv('todo_uno.csv', index=False)
tabla_202106['Predicted'] = 0
tabla_202106[['numero_de_cliente', 'Predicted']].to_csv('todo_cero.csv', index=False)

#%%


# Seleccionar las columnas que comienzan con 'Total_'
total_columns = tabla_202104.select([col for col in tabla_202104.columns if col.startswith("Total_")])


# Convertimos a un DataFrame de pandas para usar seaborn
total_df = total_columns.to_pandas()

# Crear el pairplot usando seaborn
# sns.pairplot(total_df)
# plt.show()

#%%


model.fit(total_df, y)

#%%



#%%

caract = tabla_202104.describe().to_pandas()
