#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:56:24 2025

@author: tobyh1
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import magic
#import STAGATE
import umap
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Set R paths (for mclust if used later)
os.environ['R_HOME'] = 'D:\\Program Files\\R\\R-4.0.3'
os.environ['R_USER'] = 'D:\\ProgramData\\Anaconda3\\Lib\\site-packages\\rpy2'

# Section info
section_id = '151676'
txt_path = f'Data/{section_id}/spatial/tissue_positions_list.txt'
csv_path = f'Data/{section_id}/spatial/tissue_positions_list.csv'

# Convert to CSV if needed
df = pd.read_csv(txt_path, sep=r'\s+|,', engine='python', header=None)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path, index=False, header=False)

# Load data
input_dir = os.path.join('Data', section_id)
adata = sc.read_visium(path=input_dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

# Preprocessing
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Imputation
adata.X = magic.MAGIC().fit_transform(adata.X)

# Load annotations
ann_df = pd.read_csv(f'Data/{section_id}/{section_id}_ground_truth.txt', sep='\t', header=None, index_col=0)
ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'Ground Truth']

# Plot ground truth
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])

# Run PCA
adata.obsm['X_pca'] = PCA(n_components=50).fit_transform(adata.X)

# Use spatial coordinates to define neighbors (not PCA or gene expression)
from sklearn.neighbors import NearestNeighbors

coords = adata.obsm['spatial']
n_neighbors = 8  # change as needed, e.g., 8 for adjacent spots

# Compute k-nearest neighbors on spatial coordinates
nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Build sparse connectivity matrix
from scipy.sparse import lil_matrix

n_cells = adata.n_obs
W = lil_matrix((n_cells, n_cells))

for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:  # skip the first neighbor (itself)
        W[i, j] = 1
        W[j, i] = 1  # symmetric

W = W.tocsr()  # convert to efficient format

# This replaces STAGATE's W


# Smoothing function
def weighted_sum(W, w):
    identity = scipy.sparse.identity(W.shape[0])
    W_filtered = 1 * identity + w * W

    row_sums = np.array(W_filtered.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1  # avoid division by zero

    W_filtered = scipy.sparse.diags(1 / row_sums) @ W_filtered
    smoothed = W_filtered @ adata.X
    adata.layers["smoothed"] = np.nan_to_num(smoothed, nan=0.0)
    return adata.layers["smoothed"]

# Grid search over smoothing weights
best_ari = 0
best_weight = 0
best_labels = None
# Special case: no smoothing (w = 0)
# Special case: no smoothing (w = 0)
smoothed_w0 = weighted_sum(W, 0)
adata.X = np.nan_to_num(smoothed_w0)  # Update adata.X to the w=0 smoothed version
adata.obsm['X_pca'] = PCA(n_components=50).fit_transform(adata.X)

sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)  # Now this fills adata.obsm['X_umap']

# Now cluster
labels_w0 = KMeans(n_clusters=7, random_state=0).fit_predict(adata.obsm['X_umap']).astype(str)
adata.obs['kmeans_w0'] = labels_w0

# Save into adata.obs if you want
adata.obs['kmeans_w0'] = labels_w0

for w in 0.1 * np.arange(0,10):
    smoothed = weighted_sum(W, w)
    X_dense = np.nan_to_num(adata.layers["smoothed"])
    X_pca = PCA(n_components=50).fit_transform(X_dense)
    umap_emb = umap.UMAP(n_components=20, random_state=42).fit_transform(smoothed)
    labels = KMeans(n_clusters=7, random_state=0).fit_predict(umap_emb).astype(str)
    valid_idx = ~adata.obs['Ground Truth'].isna()
    ground_truth = adata.obs['Ground Truth'][valid_idx]
    pred_labels = pd.Series(labels, index=adata.obs_names)[valid_idx]
    ari = adjusted_rand_score(ground_truth, pred_labels)

    print(f"Weight={w:.1f} | ARI={ari:.2f}")
    if ari > best_ari:
        best_ari = ari
        best_weight = w
        best_labels = labels

# Plot the w=0 result separately
sc.pl.umap(adata, color=['kmeans_w0', 'Ground Truth'], title=['KMeans (w=0)', 'Ground Truth'])
sc.pl.spatial(adata, color=['kmeans_w0', 'Ground Truth'], title=['Spatial (w=0)', 'Ground Truth'])

# Apply best clustering
adata.obs['kmeans'] = best_labels
adata.X = adata.layers['smoothed']
adata.obsm['X_pca'] = PCA(n_components=50).fit_transform(adata.X)

# Final UMAP
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)

# Evaluate and visualize
print(f'Best ARI: {best_ari:.2f} at weight {best_weight:.1f}')
sc.pl.umap(adata, color=['kmeans', 'Ground Truth'], title=[f'UMAP Clusters (ARI={best_ari:.2f})', 'Ground Truth'])
sc.pl.spatial(adata, color=['kmeans', 'Ground Truth'], title=[f'Spatial (ARI={best_ari:.2f})', 'Ground Truth'])
