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

resultado = resultado.filter(pl.col("foto_mes") == 202103)

# [:,2:-1] es para sacar numero de cliente y foto mes que no agregan data
resultado_sinna = resultado[:,2:-1].fill_null(strategy='zero')

enfermos = resultado.filter((pl.col('clase_ternaria') != 'CONTINUA') & (pl.col('clase_ternaria').is_not_null()))

enfermos.columns
enfermos['clase_ternaria'].value_counts()

X_train_test = enfermos[:,2:-1].fill_null(strategy='zero')

# %%


um_todo = umap.UMAP(n_neighbors=20, min_dist=0.5, n_components=2, metric='euclidean')
x_umap = um_todo.fit_transform(resultado_sinna)

umap.plot.points(um_todo, labels=resultado['cliente_edad'], show_legend=False)

# %%
um_e = umap.UMAP(n_neighbors=150, min_dist=0.6, n_components=2, metric='euclidean')
x_umap_e = um_e.fit_transform(X_train_test)


umap.plot.points(um_e, labels=enfermos['cliente_edad'], show_legend=False)

# %%

enfermos_label = resultado.select((pl.col("clase_ternaria") != "CONTINUA") & pl.col("clase_ternaria").is_not_null())

x_umap[:,0][enfermos_label.to_numpy()[:,0]]

enfermos_label.to_numpy()[:,0]

# %%

def pruebo_clusters(x_umap, resultado):
    for k in range(2,13):
        kmeans_centers = KMeans(n_clusters=k, random_state=10)
        kmeans_centers.fit(x_umap)
        centers = kmeans_centers.cluster_centers_
        
        
        fig, ax = plt.subplots()
        # 2nd Plot showing the actual clusters formed
        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax.scatter( x_umap[:, 0], x_umap[:, 1], marker=".", s=10, lw=0, alpha=0.4,
                   c=kmeans_centers.labels_, edgecolor="k"
                )
    
        # ax.scatter( x_umap[:, 0][enfermos_label.to_numpy()[:,0]], x_umap[:, 1][enfermos_label.to_numpy()[:,0]],
        #            marker=".", s=30, lw=0, alpha=0.7, c=kmeans_centers.labels_[enfermos_label.to_numpy()[:,0]], edgecolor="k"
        #         )
    
        # Draw white circles at cluster centers
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=100,
            edgecolor="k",
        )
    
        for i, c in enumerate(centers):
            ax.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=30, edgecolor="k")
        
        ax.set_xticks([])
        ax.set_yticks([])
    


pruebo_clusters(x_umap_e, enfermos)


# %%
for columna in X_train_test.columns:
    plt.figure(figsize=(12,7))
    plt.scatter(x_umap_e[:,0], x_umap_e[:,1], c=enfermos[columna],
                s=2, alpha=0.6)
    plt.title(columna)
    plt.colorbar()
    plt.show()





# %%

# reduce the amount of dimensions in the feature vector
pca = PCA(random_state=22)
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


# Obtenemos los componentes principales
components = pca.components_

# Crear un DataFrame para los componentes principales
feature_importances = pd.DataFrame(
    data=np.abs(components),
    columns=X_train_test.columns,
    index=[f"PC{i+1}" for i in range(len(X_train_test.columns))]
)

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

