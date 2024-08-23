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
tabla_202104 = resultado.filter(pl.col("foto_mes") == 202104)


#%%

X = tabla_202104.to_pandas().drop('clase_ternaria', axis=1)
y = tabla_202104.to_pandas()['clase_ternaria']
print(y.value_counts(normalize=True)*100)


resultado_filtrado = tabla_202104.filter(pl.col("clase_ternaria").is_null())


#%%


model = DecisionTreeClassifier(criterion='gini',
                               random_state=17,
                               min_samples_split=2000,
                               min_samples_leaf=1000,
                               max_leaf_nodes=10,
                               max_depth=5,
                               class_weight=None,
                               min_impurity_decrease=0.0001)

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

# Creando el DataFrame
df_nodos = pd.DataFrame(data, columns=['Nodo', 'Predicción', 'BAJA+1', 'BAJA+2', 'CONTINUA'])

df_nodos['ganancia1'] = (df_nodos['BAJA+2']*273000)
df_nodos['ganancia2'] = (-df_nodos['BAJA+1']*7000)
df_nodos['ganancia3'] = (-df_nodos['CONTINUA']*7000)
df_nodos['ganancia'] = df_nodos['ganancia1']+df_nodos['ganancia2']+df_nodos['ganancia3']

# Mostrando el DataFrame
print(df_nodos)

#%%


tabla_202106 = resultado.filter(pl.col("foto_mes") == 202106)

tabla_202106 = tabla_202106.to_pandas()

pred = tabla_202106['ctrx_quarter'] <= 12.5
tabla_202106['Predicted'] = pred.astype(int)

tabla_202106[['numero_de_cliente', 'Predicted']].to_csv('pred.csv', index=False)
tabla_202106['Predicted'] = 1
tabla_202106[['numero_de_cliente', 'Predicted']].to_csv('todo_uno.csv', index=False)
tabla_202106['Predicted'] = 0
tabla_202106[['numero_de_cliente', 'Predicted']].to_csv('todo_cero.csv', index=False)

#%%
