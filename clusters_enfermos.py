# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:30:44 2024

@author: jfgonzalez
"""


import polars as pl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# para UMAP
import umap
import umap.plot


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%%


resultado = pl.read_parquet('resultado_con_clase.parquet')

resultado = resultado.filter(pl.col("foto_mes") == 202104)

resultado_sinna = resultado[:,:-1].fill_null(strategy='zero')

enfermos = resultado.filter((pl.col('clase_ternaria') != 'CONTINUA') & (pl.col('clase_ternaria').is_not_null()))

enfermos.columns
enfermos['clase_ternaria'].value_counts()

X_train_test = enfermos[:,:-1].fill_null(strategy='zero')

# %%


um_todo = umap.UMAP(n_neighbors=100, min_dist=0.5, n_components=2, metric='euclidean')
x_umap = um_todo.fit_transform(resultado_sinna)

umap.plot.points(um_todo, labels=resultado['cliente_edad'], show_legend=False)

# %%
um = umap.UMAP(n_neighbors=100, min_dist=0.5, n_components=2, metric='euclidean')
x_umap = um.fit_transform(X_train_test)


umap.plot.points(um, labels=enfermos['cliente_edad'], show_legend=False)


# %%

for k in range(2,13):
    kmeans_centers = KMeans(n_clusters=k, random_state=10)
    kmeans_centers.fit(x_umap)
    centers = kmeans_centers.cluster_centers_
    
    
    fig, ax = plt.subplots()
    # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax.scatter( x_umap[:, 0], x_umap[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=kmeans_centers.labels_, edgecolor="k"
            )

    # Draw white circles at cluster centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")






# %%
for columna in X_train_test.columns:
    plt.figure(figsize=(7,7))
    plt.scatter(x_umap[:,0], x_umap[:,1], c=enfermos[columna], alpha=0.5)
    plt.title(columna)
    plt.colorbar()
    plt.show()


# %%

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=10, random_state=22)
pca.fit(X_train_test)

x_pca = pca.transform(X_train_test)

# %%
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(x_pca[:,0],x_pca[:,1], alpha=0.7)
plt.colorbar();
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.subplot(1,2,2)
plt.scatter(x_pca[:,2],x_pca[:,3], alpha=0.7)
plt.colorbar(ticks=range(6));
plt.xlabel('PC3')
plt.ylabel('PC4')

# %% 

sns.pairplot(pd.DataFrame(x_pca))



# %%

vanDongen_table = []
Rand_table = []
n_clusters_table = range(2,10)

for i in n_clusters_table:
  kmeans_temp = KMeans(n_clusters=i)
  kmeans_temp.fit(x_umap)
  tmp_table = pd.DataFrame({'Predict': kmeans_temp.labels_, 'Real': org_lab})
  ct = pd.crosstab(tmp_table['Predict'], tmp_table['Real'])
  vanDongen_table.append(vanDongen(ct))
  Rand_table.append(adjusted_rand_score(kmeans_temp.labels_, org_lab))
  
pd.DataFrame({'vanDongen': vanDongen_table, 'Rand': Rand_table}, index=n_clusters_table)

# %%


umap.plot.points(um, labels=org_lab)

