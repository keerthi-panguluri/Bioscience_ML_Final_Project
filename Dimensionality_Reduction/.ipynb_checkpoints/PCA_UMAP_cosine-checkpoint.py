# 
# PCA & UMAP (Cosine)
#
# mclust -> ARI = 0.27
# k-means -> ARI = 0.23
# PCA -> sc.pp.pca(adata, n_comps=30)
# Cosine
# UMAP
# using both PCA and UMAP b/c:
    # PCA reduces dimensionality and noise (e.g., 30 dimensions).
    # UMAP takes the PCA result and projects it into 2D space for visualization while 
    # preserving relationships.
    # The final 2D coordinates are used for plotting, clustering (e.g., Mclust), and 
    # interpretation.

#---------------------WHY COSINE DID NOT IMPROVE ARI-------------------------------#
# Cosine isn’t automatically better — it's context-dependent.
# Mclust prefers Gaussian-shaped clusters, which cosine UMAP might not preserve.
# Try comparing ARIs between Euclidean and cosine using both PCA directly and 
# UMAP outputs to understand what’s affecting your clustering.

# Imports and setup
import os
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
import umap
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# R and rpy2 for Mclust
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
mclust = importr('mclust')

# R setup (edit paths if needed for your system)
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.environ['R_USER'] = '/opt/anaconda3/envs/tf_env/lib/python3.10/site-packages/rpy2'

warnings.filterwarnings("ignore")

# Load data
section_id = '151676'
input_dir = os.path.join('Data', section_id)
adata = sc.read_visium(path=input_dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

# Preprocess
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Load ground truth
ann_df = pd.read_csv(os.path.join('Data', section_id, f'{section_id}_ground_truth.txt'),
                     sep='\t', header=None, index_col=0)
ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'Ground Truth']

# PCA
sc.pp.pca(adata, n_comps=30)

# UMAP using cosine distance on PCA embeddings
reducer = umap.UMAP(n_neighbors=15, metric='cosine', random_state=42)
adata.obsm['X_umap_cosine'] = reducer.fit_transform(adata.obsm['X_pca'])

# Combine PCA and UMAP embeddings into one feature space
combined_embeddings = np.hstack([adata.obsm['X_pca'], adata.obsm['X_umap_cosine']])
adata.obsm['X_combined'] = combined_embeddings

# Mclust clustering on combined PCA and UMAP embeddings
combined_df = pd.DataFrame(combined_embeddings, index=adata.obs_names)
ro.globalenv['combined_data'] = pandas2ri.py2rpy(combined_df)
ro.r('mclust_result_combined <- Mclust(combined_data, G=7)')
adata.obs['mclust_combined'] = list(ro.r('mclust_result_combined$classification'))

# K-means clustering on combined PCA and UMAP embeddings (7 clusters)
kmeans = KMeans(n_clusters=7, random_state=42)
adata.obs['kmeans_combined'] = kmeans.fit_predict(combined_embeddings)

# ARI evaluation for both Mclust and K-means
obs_df = adata.obs.dropna()
ari_combined_mclust = adjusted_rand_score(obs_df['mclust_combined'], obs_df['Ground Truth'])
ari_combined_kmeans = adjusted_rand_score(obs_df['kmeans_combined'], obs_df['Ground Truth'])

print(f'Adjusted Rand Index (Mclust, combined): {ari_combined_mclust:.2f}')
print(f'Adjusted Rand Index (K-means, combined): {ari_combined_kmeans:.2f}')

# Visualization: spatial + UMAP for both Mclust and K-means
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.embedding(adata, basis='X_umap_cosine', color=['mclust_combined', 'kmeans_combined', 'Ground Truth'],
                title=[f'Mclust Combined (ARI={ari_combined_mclust:.2f})', f'K-means Combined (ARI={ari_combined_kmeans:.2f})', 'Ground Truth'])

sc.pl.spatial(adata, color=["mclust_combined", "kmeans_combined", "Ground Truth"],
              title=[f'Mclust Combined (ARI={ari_combined_mclust:.2f})', f'K-means Combined (ARI={ari_combined_kmeans:.2f})', "Ground Truth"])

# Optional: PAGA on annotated data
used_adata = adata[adata.obs['Ground Truth'] != 'nan', :]
sc.pp.neighbors(used_adata, n_neighbors=15, use_rep='X_combined')

sc.tl.paga(used_adata, groups='Ground Truth')
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20, 
                   title=f'{section_id}_Mclust_Kmeans_Combined', legend_fontoutline=2, show=False)
