# 
# tSNE & PCA
#
# PCA = 30, preplexity = 50
# mclust -> ARI = 0.30
# k-means -> ARI = 0.27
#
# PCA = 30, preplexity = 30
# mclust -> ARI = 0.30
# k-means -> ARI = 0.27
#
# PCA = 30, preplexity = 40
# mclust -> ARI = 0.28
# k-means -> ARI = 0.29


import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA  

# R and rpy2 for Mclust
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
mclust = importr('mclust')

# Set environment variables for R compatibility
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.environ['R_USER'] = '/opt/anaconda3/envs/tf_env/lib/python3.10/site-packages/rpy2'

warnings.filterwarnings("ignore")

# Section ID and data path
section_id = '151676'
input_dir = os.path.join('Data', section_id)

# Load Visium spatial transcriptomics data
adata = sc.read_visium(path=input_dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

# Preprocessing
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Load ground truth annotations
ann_df = pd.read_csv(os.path.join('Data', section_id, f'{section_id}_ground_truth.txt'), sep='\t', header=None, index_col=0)
ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'Ground Truth']

# Visualize ground truth
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])

# PCA (50 components)
sc.tl.pca(adata, n_comps=30)
print("PCA explained variance ratio:", adata.uns['pca']['variance_ratio'])

# t-SNE on PCA
sc.tl.tsne(adata, n_pcs=30, perplexity=40, random_state=42)

# Combine PCA and t-SNE
combined_features = np.concatenate([adata.obsm['X_pca'], adata.obsm['X_tsne']], axis=1)
combined_df = pd.DataFrame(combined_features, index=adata.obs_names)

# Mclust on PCA + t-SNE
ro.globalenv['combined_data'] = pandas2ri.py2rpy(combined_df)
ro.r('set.seed(42)')
ro.r('mclust_result <- Mclust(combined_data, G=7)')
adata.obs['mclust'] = list(ro.r('mclust_result$classification'))

# KMeans on PCA + t-SNE
kmeans = KMeans(n_clusters=7, random_state=42)
adata.obs['kmeans'] = kmeans.fit_predict(combined_df)

# Filter for non-NaN
obs_df = adata.obs.dropna(subset=['mclust', 'Ground Truth'])

# ARI evaluation
ari_mclust = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
ari_kmeans = adjusted_rand_score(obs_df['kmeans'], obs_df['Ground Truth'])
print(f'Adjusted Rand Index (Mclust PCA+tSNE): {ari_mclust:.2f}')
print(f'Adjusted Rand Index (KMeans PCA+tSNE): {ari_kmeans:.2f}')

# t-SNE visualizations
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.embedding(adata, basis='tsne', color=["mclust", "kmeans", "Ground Truth"],
                title=[f'Mclust (ARI={ari_mclust:.2f})', f'KMeans (ARI={ari_kmeans:.2f})', "Ground Truth"])

# Spatial visualizations
sc.pl.spatial(adata, color=["mclust", "kmeans", "Ground Truth"],
              title=[f'Mclust (ARI={ari_mclust:.2f})', f'KMeans (ARI={ari_kmeans:.2f})', "Ground Truth"])

# PAGA
used_adata = adata[adata.obs['Ground Truth'] != 'nan', :]
sc.tl.paga(used_adata, groups='Ground Truth')
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                   title=f'{section_id}_Clustering_PCA_tSNE', legend_fontoutline=2, show=False)
