# UMAP only:
# mclust -> ARI = 0.28
# k-means -> ARI = 0.31
# 

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

# R and rpy2 for Mclust
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
mclust = importr('mclust')

# Set environment variables for R compatibility (adjust paths as needed)
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

# PCA + UMAP for visualization
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(adata, min_dist=0.3)

# Mclust clustering using UMAP coordinates
umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs_names)
ro.globalenv['umap_data'] = pandas2ri.py2rpy(umap_df)
ro.r('mclust_result <- Mclust(umap_data, G=7)')
adata.obs['mclust'] = list(ro.r('mclust_result$classification'))

# K-means clustering using UMAP coordinates
kmeans = KMeans(n_clusters=7, random_state=42)
adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_umap'])

# ARI evaluation for Mclust and K-means
obs_df = adata.obs.dropna()
ari_mclust = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
ari_kmeans = adjusted_rand_score(obs_df['kmeans'], obs_df['Ground Truth'])

print(f'Mclust Adjusted Rand Index (UMAP): {ari_mclust:.2f}')
print(f'K-means Adjusted Rand Index (UMAP): {ari_kmeans:.2f}')

# UMAP plots
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.umap(adata, color=["mclust", "kmeans", "Ground Truth"],
           title=[f'Mclust (ARI={ari_mclust:.2f})', f'K-means (ARI={ari_kmeans:.2f})', "Ground Truth"])

# Spatial plots
sc.pl.spatial(adata, color=["mclust", "kmeans", "Ground Truth"],
              title=[f'Mclust (ARI={ari_mclust:.2f})', f'K-means (ARI={ari_kmeans:.2f})', "Ground Truth"])

# PAGA analysis using Ground Truth
used_adata = adata[adata.obs['Ground Truth'] != 'nan', :]
sc.tl.paga(used_adata, groups='Ground Truth')
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                   title=f'{section_id}_Mclust_Kmeans_UMAP', legend_fontoutline=2, show=False)
