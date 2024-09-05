# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 08:36:33 2024

@author: jfgonzalez
"""


# %%
# 799891, 799921, 799961, 799991, 800011

import polars as pl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler

# %%


def plot_histogram_with_bin_width(Predicted, y_max=40000, x_max=0.4,
                                  last_n_cases=10000, bin_width=0.001):
    """
    Genera un histograma de las probabilidades con un ancho de bin específico,
    límites en los ejes, y una línea vertical que marca el valor de x
    correspondiente a los últimos n casos. La escala del eje x es logarítmica.

    Parameters:
    - Predicted: Array de probabilidades predichas (ej. Predicted[:, 1]).
    - y_max: Límite máximo del eje Y (default: 40000).
    - x_max: Límite máximo del eje X (default: 0.4).
    - last_n_cases: Número de últimos casos para marcar en el histograma.
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
    plt.axvline(x=x_value, color='red', linestyle='--',
                label=f'Últimos {last_n_cases} casos en x = {x_value:.4f}')

    plt.axvline(x=1/40, color='blue', linestyle='--', label=f'x = {1/40:.4f}')
    # Añadir una etiqueta a la línea
    plt.text(x_value * 1.1, y_max * 0.5, f'x = {x_value:.4f}',
             rotation=90, verticalalignment='center')

    # Añadir títulos y etiquetas
    plt.title('Distribución de Predicted[:, 1]')
    plt.xlabel('Probabilidad (Escala Logarítmica)')
    plt.ylabel('Frecuencia')
    plt.legend()

    # Mostrar el gráfico
    plt.show()


# %% uno los datos de tarjetas


resultado = pl.read_parquet('resultado_con_clase.parquet')

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
    resultado = resultado.with_columns(
        (pl.col(f'Master{suffix}') +
         pl.col(f'Visa{suffix}')).alias(f'Total{suffix}')
    )


tabla_202104 = resultado.filter(pl.col("foto_mes") <= 202104)
tabla_202106 = resultado.filter(pl.col("foto_mes") == 202106)
# Ahora df tiene columnas sumadas por cada sufijo

# %%

X = tabla_202104.to_pandas().drop('clase_ternaria', axis=1)
y = tabla_202104.to_pandas()['clase_ternaria']
print(y.value_counts(normalize=True)*100)


# resultado_filtrado = tabla_202104.filter(pl.col("clase_ternaria").is_null())


# %%


model = DecisionTreeClassifier(criterion='gini',
                               random_state=17,
                               min_samples_split=250,
                               min_samples_leaf=100,
                               max_leaf_nodes=16,
                               max_depth=7,
                               class_weight=None,
                               min_impurity_decrease=0.000)

model.fit(X, y)

features = pd.DataFrame([model.feature_names_in_,
                         model.feature_importances_]).T


# %% Visualización

# Todo es mucho más bonito si tiene colores
def dibujo_arbol(model, X):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, filled=True,
              class_names=model.classes_, rounded=True,
              impurity=True, fontsize=8,  proportion=False,
              node_ids=True, )
    plt.show()


dibujo_arbol(model, X)

# %% calcular_ganancia_polars


def calcular_ganancia_polars(model, corte):
    # Obteniendo el árbol y los valores asociados
    tree = model.tree_
    nodos = tree.node_count
    valores = tree.value
    pesos = tree.weighted_n_node_samples

    # Creando una lista para almacenar los datos de cada nodo
    data = []

    for i in range(nodos):
        # Para cada nodo, obtenemos la predicción (clase mayoritaria)
        # y los valores de cada clase multiplicados por los pesos
        prediccion = np.argmax(valores[i])
        clase_0 = int(valores[i][0][0] * pesos[i])
        clase_1 = int(valores[i][0][1] * pesos[i])
        clase_2 = int(valores[i][0][2] * pesos[i])

        data.append([i, prediccion, clase_0, clase_1, clase_2])

    # Creando el DataFrame en Polars
    df_nodos = pl.DataFrame(
        data, orient="row",
        schema=['Nodo', 'Predicción', 'BAJA+1', 'BAJA+2', 'CONTINUA']
    )

    # Calcular las ganancias
    df_nodos = df_nodos.with_columns([
        (pl.col('BAJA+2') * 273000).alias('ganancia1'),
        (-pl.col('BAJA+1') * 7000).alias('ganancia2'),
        (-pl.col('CONTINUA') * 7000).alias('ganancia3'),
        (pl.col('BAJA+1') + pl.col('CONTINUA')).alias('Otros'),
        (pl.col('BAJA+2') * 273000 - pl.col('BAJA+1') * 7000 -
         pl.col('CONTINUA') * 7000).alias('ganancia'),
        ((pl.col('BAJA+2') / (pl.col('BAJA+1') + pl.col('BAJA+2') +
                              pl.col('CONTINUA'))) * 100).alias('prob_+2')
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

    df_resultado = df_hojas.select(['Otros', 'BAJA+2', 'ganancia', 'prob_+2'
                                    ]).sort('prob_+2', descending=True)
    filtered_sum = df_resultado.filter(pl.col("prob_+2") > corte
                                       ).select(pl.col("ganancia").sum())
    filtered_sum_pond = filtered_sum/(0.75*4)

    df_resultado_final = pl.DataFrame({
        "ganancia_positiva": filtered_sum,
        "ganancia_ponderada": filtered_sum_pond
    })

    with pl.Config(
        tbl_cell_numeric_alignment="RIGHT",
        thousands_separator=True,
        float_precision=2,
        tbl_rows=20
    ):
        print(df_resultado)
        print(df_resultado_final)

    return df_resultado


df_resultado = calcular_ganancia_polars(model, 2.5)


# %%

# Supongamos que X son tus características e y es tu variable objetivo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
# %%

model.fit(X_train, y_train)

# %%

Predicted = model.predict_proba(X_test, )
por_proba = Predicted[:, 1] > 1/40
test_clase = (y_test == 'BAJA+2')


def chequeo_gt(por_proba, test_clase):
    df = pl.DataFrame({
        "por_proba": por_proba,
        "test_clase": test_clase
    })

    # Agregamos la tercer columna con las condiciones
    df = df.with_columns(
        pl.when((pl.col("por_proba")) & (pl.col("test_clase")))
          .then(273000)
          .when((pl.col("por_proba")) & (~pl.col("test_clase")))
          .then(-7000)
          .otherwise(0)
          .alias("resultado")
    )
    suma_resultado = df.select(pl.col("resultado").sum()).item()
    print(df)
    print(f'{suma_resultado:,.0f}')

    return suma_resultado


chequeo = chequeo_gt(por_proba, test_clase)

# %%


def metrica_ganancia(estimator, X, y):
    # Predecir las probabilidades usando el estimador proporcionado
    Predicted = estimator.predict_proba(X)

    # Definir la condición basada en la probabilidad
    por_proba = Predicted[:, 1] > 1/40

    # Verificar si la clase real es 'BAJA+2'
    test_clase = y == 'BAJA+2'

    # Crear DataFrame con Polars
    df = pl.DataFrame({
        "por_proba": por_proba,
        "test_clase": test_clase
    })

    # Calcular la ganancia usando condiciones
    df = df.with_columns(
        pl.when((pl.col("por_proba")) & (pl.col("test_clase")))
          .then(273000)
          .when((pl.col("por_proba")) & (~pl.col("test_clase")))
          .then(-7000)
          .otherwise(0)
          .alias("resultado")
    )

    # Sumar el resultado
    suma_resultado = df.select(pl.col("resultado").sum()).item()

    return suma_resultado


# %%
# 799891, 799921, 799961, 799991, 800011

# Hiperparámetros a buscar
param_dist = {
    'criterion': ['gini'],
    'min_samples_split': np.arange(200, 300, 10),
    'min_samples_leaf': np.arange(50, 150, 10),
    'max_leaf_nodes': np.arange(10, 20, 1),
    'max_depth': np.arange(5, 15, 1)
}

param_list = list(ParameterSampler(param_distributions=param_dist,
                                   n_iter=50, random_state=42))
# %%

# Lista para almacenar los resultados
resultados = []

for i, params in enumerate(param_list):
    # Crear el modelo con los parámetros actuales
    model = DecisionTreeClassifier(random_state=17, **params)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    calcular_ganancia_polars(model, 2.5)

    ganancia = metrica_ganancia(model, X_test, y_test)

    # Guardar los resultados
    resultados.append({'params': params, 'ganancia': ganancia})

    print(f"Modelo {i+1} evaluado con ganancia: {ganancia}")

# Ordenar los resultados por ganancia
resultados_ordenados = sorted(resultados, key=lambda x: x['ganancia'],
                              reverse=True)

# Mostrar los mejores resultados
for res in resultados_ordenados[:5]:  # Mostrar los top 5
    print(f"Parámetros: {res['params']}, Ganancia: {res['ganancia']}")

# %%

# Primos 799891, 799921, 799961, 799991, 800011
primos = [799891, 799921, 799961, 799991, 800011]

# Hiperparámetros a buscar
param_dist = {
    'criterion': ['gini'],
    'min_samples_split': np.arange(200, 300, 10),
    'min_samples_leaf': np.arange(50, 150, 10),
    'max_leaf_nodes': np.arange(10, 20, 1),
    'max_depth': np.arange(5, 15, 1)
}

param_list = list(ParameterSampler(param_distributions=param_dist,
                                   n_iter=5, random_state=42))

# Lista para almacenar los resultados
resultados = []

for i, params in enumerate(param_list):
    ganancias = []

    for seed in primos:
        # Crear el modelo con los parámetros actuales y la semilla específica
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed)

        model = DecisionTreeClassifier(random_state=799891, **params)

        # Entrenar el modelo
        model.fit(X_train, y_train)

        calcular_ganancia_polars(model, 2.5)

        # Calcular la ganancia
        ganancia = metrica_ganancia(model, X_test, y_test)
        ganancias.append(ganancia)

    # Calcular el promedio de las ganancias para las 5 semillas
    ganancia_promedio = np.mean(ganancias)

    # Guardar los resultados
    resultados.append({**params,
                       'ganancia_promedio': ganancia_promedio})

    print(f"Modelo {i+1} evaluado con ganancia promedio: {ganancia_promedio}")

df_resultados = pd.DataFrame(resultados)


# Ordenar los resultados por ganancia promedio en orden descendente
df_resultados_ordenados = df_resultados.sort_values(by='ganancia_promedio',
                                                    ascending=False)

# Mostrar los mejores resultados (top 5)
print(df_resultados_ordenados.head(5))

df_resultados_ordenados.to_csv('parametros_y_resultados.csv')

# %%


# %%


params = {'min_samples_split': 290, 'min_samples_leaf': 140,
          'max_leaf_nodes': 18, 'max_depth': 6, 'criterion': 'gini'}

model = DecisionTreeClassifier(random_state=17, **params)

# Entrenar el modelo
model.fit(X_train, y_train)

calcular_ganancia_polars(model, 2.5)

ganancia = metrica_ganancia(model, X_test, y_test)

# Ganancia: 87577000

# %%

# Graficar la importancia de las características
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]


# Asumiendo que 'importances' y 'indices' ya están definidos
# Restricción a las 10 columnas más importantes
top_n = 15
top_indices = indices[:top_n]

plt.figure(figsize=(10, 6))
plt.title("Importancia de las características")
plt.bar(range(top_n), importances[top_indices], align="center")
plt.xticks(range(top_n), [X.columns[i] for i in top_indices], rotation=90)
plt.show()


# %%

caract = tabla_202104.describe().to_pandas()

# %%
Xa = tabla_202106.to_pandas().drop('clase_ternaria', axis=1)

Predicted = model.predict_proba(Xa)
por_proba = Predicted[:, 1] > 1/40

entrega = pd.DataFrame([Xa.numero_de_cliente, por_proba]).T
entrega.columns = ['numero_de_cliente', 'Predicted']
print(entrega.Predicted.value_counts())
entrega['Predicted'] = entrega['Predicted'].astype(int)
print(entrega.Predicted.value_counts())
entrega[['numero_de_cliente', 'Predicted']].to_csv('./pred.csv', index=False)


# %%
