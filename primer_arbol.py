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


# %%



def plot_histogram_with_bin_width(Predicted, y_max=40000, x_max=0.4, last_n_cases=10000, bin_width=0.001):
    """
    Genera un histograma de las probabilidades con un ancho de bin específico, límites en los ejes, y una línea vertical que marca
    el valor de x correspondiente a los últimos n casos. La escala del eje x es logarítmica.

    Parameters:
    - Predicted: Array de probabilidades predichas (ej. Predicted[:, 1]).
    - y_max: Límite máximo del eje Y (default: 40000).
    - x_max: Límite máximo del eje X (default: 0.4).
    - last_n_cases: Número de últimos casos para marcar en el histograma (default: 10000).
    - bin_width: Ancho de cada bin en el histograma (default: 0.001).
    """
    # Definir los bordes de los bins basados en el ancho especificado
    bins = np.arange(0, x_max + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    ax = plt.hist(Predicted, bins=bins, color='skyblue', edgecolor='black')

    # Establecer los límites en los ejes
    plt.ylim(0, y_max)  # Límite máximo en el eje Y
    plt.xlim(left=1e-4, right=x_max)  # Límite máximo en el eje X

    # Establecer la escala logarítmica en el eje X

    # Calcular el valor de x para los últimos n casos
    cumsum = np.cumsum(ax[0][::-1])  # Acumulado desde la derecha
    x_value = ax[1][::-1][np.argmax(cumsum > last_n_cases)]  # Valor de x

    # Dibujar una línea vertical en el valor de x correspondiente
    plt.axvline(x=x_value, color='red', linestyle='--', label=f'Últimos {last_n_cases} casos en x = {x_value:.4f}')

    plt.axvline(x=1/40, color='blue', linestyle='--', label=f'x = {1/40:.4f}')
    # Añadir una etiqueta a la línea
    plt.text(x_value * 1.1, y_max * 0.5, f'x = {x_value:.4f}', rotation=90, verticalalignment='center')

    # Añadir títulos y etiquetas
    plt.title('Distribución de Predicted[:, 1]')
    plt.xlabel('Probabilidad (Escala Logarítmica)')
    plt.ylabel('Frecuencia')
    plt.legend()

    # Mostrar el gráfico
    plt.show()


#%% uno los datos de tarjetas

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
def dibujo_arbol(model,X):
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=X.columns, filled=True,
              class_names=model.classes_, rounded=True,
              impurity=True, fontsize=8,  proportion=False,
              node_ids=True, )
    plt.show()

dibujo_arbol(model,X)

#%%


def calcular_ganancia_polars(model, corte):
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
    filtered_sum = df_resultado.filter(pl.col("prob_+2") > corte).select(pl.col("ganancia").sum())
    filtered_sum_pond = filtered_sum/(0.75*4)

    with pl.Config(
    tbl_cell_numeric_alignment="RIGHT",
    thousands_separator=True,
    float_precision=2,
    tbl_rows=20
    ):
        pl.Config.set_tbl_cell_alignment("RIGHT")
        print(df_resultado)
        print(filtered_sum, filtered_sum_pond)
        

    return df_resultado

df_resultado = calcular_ganancia_polars(model, 2.5)


#%%


from sklearn.model_selection import train_test_split

# Supongamos que X son tus características e y es tu variable objetivo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
#%%
model.fit(X_train, y_train)

df_resultado = calcular_ganancia_polars(model, 2.5)

#%%

Predicted = model.predict_proba(X_test, )

plot_histogram_with_bin_width(Predicted[:, 1], y_max=40000, x_max=0.1, last_n_cases=10000, bin_width=0.001)

#%%
por_proba = Predicted[:,1]>1/40

test_clase = (y_test == 'BAJA+2') 

def chequeo_gt(por_proba, test_clase):
    df = pl.DataFrame({
        "por_proba": por_proba,
        "test_clase": test_clase
    })
    
    # Agregamos la tercer columna con las condiciones
    df = df.with_columns(
        pl.when((pl.col("por_proba") == True) & (pl.col("test_clase") == True))
          .then(273000)
          .when((pl.col("por_proba") == True) & (pl.col("test_clase") == False))
          .then(-7000)
          .otherwise(0)
          .alias("resultado")
    )
    suma_resultado = df.select(pl.col("resultado").sum()).item()
    print(f'{suma_resultado:,.0f}')
    return df

chequeo = chequeo_gt(por_proba, test_clase)

#%%


tabla_202106 = resultado.filter(pl.col("foto_mes") == 202106)

tabla_202106 = tabla_202106.to_pandas()

pred = tabla_202106['ctrx_quarter'] <= 3.5
tabla_202106['Predicted'] = pred.astype(int)


#%%

Predicted = model.predict_proba(Xa, )

plot_histogram_with_bin_width(Predicted[:, 1], y_max=40000, x_max=0.1, last_n_cases=10000, bin_width=0.001)

por_proba = Predicted[:,1]>1/40
#%%

entrega = pd.DataFrame([Xa.numero_de_cliente, por_proba]).T
entrega.columns=['numero_de_cliente', 'Predicted']
print(entrega.Predicted.value_counts())
entrega['Predicted'] = entrega['Predicted'].astype(int)
print(entrega.Predicted.value_counts())
entrega[['numero_de_cliente', 'Predicted']].to_csv('./pred.csv', index=False)


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


def plot_histogram_with_bin_width(Predicted, y_max=40000, x_max=0.4, last_n_cases=10000, bin_width=0.001):
    """
    Genera un histograma de las probabilidades con un ancho de bin específico, límites en los ejes, y una línea vertical que marca
    el valor de x correspondiente a los últimos n casos. La escala del eje x es logarítmica.

    Parameters:
    - Predicted: Array de probabilidades predichas (ej. Predicted[:, 1]).
    - y_max: Límite máximo del eje Y (default: 40000).
    - x_max: Límite máximo del eje X (default: 0.4).
    - last_n_cases: Número de últimos casos para marcar en el histograma (default: 10000).
    - bin_width: Ancho de cada bin en el histograma (default: 0.001).
    """
    # Definir los bordes de los bins basados en el ancho especificado
    bins = np.arange(0, x_max + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    ax = plt.hist(Predicted, bins=bins, color='skyblue', edgecolor='black')

    # Establecer los límites en los ejes
    plt.ylim(0, y_max)  # Límite máximo en el eje Y
    plt.xlim(left=1e-4, right=x_max)  # Límite máximo en el eje X

    # Establecer la escala logarítmica en el eje X

    # Calcular el valor de x para los últimos n casos
    cumsum = np.cumsum(ax[0][::-1])  # Acumulado desde la derecha
    x_value = ax[1][::-1][np.argmax(cumsum > last_n_cases)]  # Valor de x

    # Dibujar una línea vertical en el valor de x correspondiente
    plt.axvline(x=x_value, color='red', linestyle='--', label=f'Últimos {last_n_cases} casos en x = {x_value:.4f}')

    plt.axvline(x=1/40, color='blue', linestyle='--', label=f'x = {1/40:.4f}')
    # Añadir una etiqueta a la línea
    plt.text(x_value * 1.1, y_max * 0.5, f'x = {x_value:.4f}', rotation=90, verticalalignment='center')

    # Añadir títulos y etiquetas
    plt.title('Distribución de Predicted[:, 1]')
    plt.xlabel('Probabilidad (Escala Logarítmica)')
    plt.ylabel('Frecuencia')
    plt.legend()

    # Mostrar el gráfico
    plt.show()


#%%

caract = tabla_202104.describe().to_pandas()
#%%


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Hiperparámetros a buscar
param_dist = {
    'criterion': ['gini', 'entropy'],
    'min_samples_split': np.arange(200, 300, 10),
    'min_samples_leaf': np.arange(50, 150, 10),
    'max_leaf_nodes': np.arange(10, 20, 1),
    'max_depth': np.arange(5, 15, 1),
    'min_impurity_decrease': np.linspace(0.0, 0.01, 10)
}

# Configurar el RandomizedSearchCV
clf = DecisionTreeClassifier(random_state=17)
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=50, cv=5, random_state=42, n_jobs=-1)

# Separar los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Ajustar el modelo
random_search.fit(X_train, y_train)

#%%

# Mejor modelo encontrado
best_model = random_search.best_estimator_

# Predecir en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Best model accuracy: {accuracy:.4f}")

# Graficar la matriz de confusión
# plot_confusion_matrix(best_model, X_test, y_test)
# plt.title("Matriz de Confusión del Mejor Modelo")
# plt.show()

# Graficar la matriz de confusión
cm_display = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
cm_display.ax_.set_title("Matriz de Confusión del Mejor Modelo")
plt.show()

#%%

# Graficar la importancia de las características
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Importancia de las características")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.show()

# Graficar la precisión del modelo en función de los hiperparámetros
results = random_search.cv_results_

plt.figure(figsize=(10, 6))
plt.plot(results['mean_test_score'])
plt.title("Precisión media del modelo a través de los hiperparámetros")
plt.xlabel("Iteraciones")
plt.ylabel("Precisión media")
plt.show()


#%%