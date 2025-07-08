# 
# PCA & UMAP (Euclidean)
#
# mclust -> ARI = 0.34
# k-means -> ARI = 0.27
#
# PCA -> sc.pp.pca(adata, n_comps=30)
# Euclidean Distance -> sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca') 
# UMAP -> sc.tl.umap(adata, min_dist = 0.3)
# using both PCA and UMAP b/c:
    # PCA reduces dimensionality and noise (e.g., 30 dimensions).
    # UMAP takes the PCA result and projects it into 2D space for visualization while 
    # preserving relationships.
    # The final 2D coordinates are used for plotting, clustering (e.g., Mclust), and 
    # interpretation.

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

# PCA + UMAP
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(adata, min_dist=0.3)

# Combine PCA + UMAP for clustering
combined_features = np.concatenate((adata.obsm['X_pca'], adata.obsm['X_umap']), axis=1)
combined_df = pd.DataFrame(combined_features, index=adata.obs_names)

# Mclust on combined features
ro.globalenv['combined_data'] = pandas2ri.py2rpy(combined_df)
ro.r('mclust_result <- Mclust(combined_data, G=7)')
adata.obs['mclust'] = list(ro.r('mclust_result$classification'))

# K-means on combined features
kmeans = KMeans(n_clusters=7, random_state=42)
adata.obs['kmeans'] = kmeans.fit_predict(combined_features)

# ARI scores
obs_df = adata.obs.dropna()
ari_mclust = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
ari_kmeans = adjusted_rand_score(obs_df['kmeans'], obs_df['Ground Truth'])

print(f'Mclust Adjusted Rand Index (PCA+UMAP): {ari_mclust:.2f}')
print(f'K-means Adjusted Rand Index (PCA+UMAP): {ari_kmeans:.2f}')

# UMAP visualization
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.umap(adata, color=["mclust", "kmeans", "Ground Truth"],
           title=[f'Mclust (ARI={ari_mclust:.2f})', f'K-means (ARI={ari_kmeans:.2f})', "Ground Truth"])

# Spatial visualization
sc.pl.spatial(adata, color=["mclust", "kmeans", "Ground Truth"],
              title=[f'Mclust (ARI={ari_mclust:.2f})', f'K-means (ARI={ari_kmeans:.2f})', "Ground Truth"])

# PAGA on Ground Truth
used_adata = adata[adata.obs['Ground Truth'] != 'nan', :]
sc.tl.paga(used_adata, groups='Ground Truth')
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                   title=f'{section_id}_Clustering_PCA_UMAP', legend_fontoutline=2, show=False)
