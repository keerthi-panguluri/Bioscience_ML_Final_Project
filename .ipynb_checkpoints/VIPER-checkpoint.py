# MCLUST - Original ARI: 0.314, Smoothed ARI: 0.391
# KMEANS - Original ARI: 0.176, Smoothed ARI: 0.250

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
from scipy.spatial import KDTree

# R and rpy2 for Mclust and VIPER
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()

# Import R packages
mclust = importr('mclust')
viper = importr('viper')

# Set R environment variables
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
os.environ['R_USER'] = '/opt/anaconda3/envs/tf_env/lib/python3.10/site-packages/rpy2'

warnings.filterwarnings("ignore")

# Load spatial transcriptomics data
section_id = '151676'
input_dir = os.path.join('Data', section_id)
adata = sc.read_visium(path=input_dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

# Preprocessing
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Ground truth annotations
ann_df = pd.read_csv(os.path.join('Data', section_id, f'{section_id}_ground_truth.txt'), 
                     sep='\t', header=None, index_col=0)
ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = ann_df.loc[adata.obs_names, 'Ground Truth']

# PCA and neighbors with spatial information
sc.pp.pca(adata, n_comps=50, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=20, use_rep='X_pca', metric='cosine', n_pcs=30)
spatial_coords = adata.obsm['spatial']
adata.obsm['X_pca_spatial'] = np.concatenate([
    adata.obsm['X_pca'][:, :30],
    spatial_coords / spatial_coords.max(axis=0)
], axis=1)

# K-means clustering
kmeans = KMeans(n_clusters=7, init='k-means++', n_init=50, random_state=42)
adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca_spatial'])

# Mclust clustering in R
pca_spatial_df = pd.DataFrame(adata.obsm['X_pca_spatial'], index=adata.obs_names)
ro.globalenv['pca_spatial_data'] = pandas2ri.py2rpy(pca_spatial_df)
ro.r('''
mclust_result <- Mclust(pca_spatial_data, G=7, modelNames="VEV")
''')
adata.obs['mclust'] = list(ro.r('mclust_result$classification'))

# VIPER analysis
expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                       index=adata.obs_names,
                       columns=adata.var_names)
ro.globalenv['expr_matrix'] = pandas2ri.py2rpy(expr_df.T)
ro.r('load("/Users/keerthi/regulon.rdata")')
ro.r('''
vpres <- viper(expr_matrix, regulon, method="rank", pleiotropy=TRUE, minsize=5, verbose=FALSE)
''')
viper_scores = ro.r('as.data.frame(t(vpres))')
viper_df = pandas2ri.rpy2py(viper_scores)
adata.obsm['X_viper'] = viper_df.values

# UMAP and neighbors on VIPER results
sc.pp.neighbors(adata, use_rep='X_viper', n_neighbors=15, metric='correlation')
sc.tl.umap(adata)

# Spatial smoothing function
def spatial_smoothing(labels, coordinates, k=15):
    tree = KDTree(coordinates)
    smoothed_labels = labels.copy()
    for i in range(len(labels)):
        _, indices = tree.query(coordinates[i], k=k)
        neighbor_labels = labels.iloc[indices]
        smoothed_labels.iloc[i] = neighbor_labels.mode()[0]
    return smoothed_labels

# Apply smoothing
for method in ['mclust', 'kmeans']:
    adata.obs[f'{method}_smoothed'] = spatial_smoothing(
        adata.obs[method], 
        adata.obsm['spatial'], 
        k=15
    )

# Evaluate clustering results
obs_df = adata.obs.dropna()
results = {}
for method in ['mclust', 'kmeans']:
    orig_ari = adjusted_rand_score(obs_df[method], obs_df['Ground Truth'])
    smoothed_ari = adjusted_rand_score(obs_df[f'{method}_smoothed'], obs_df['Ground Truth'])
    results[method] = (orig_ari, smoothed_ari)
    print(f'{method.upper()} - Original ARI: {orig_ari:.3f}, Smoothed ARI: {smoothed_ari:.3f}')

# Visualization
plt.rcParams["figure.figsize"] = (8, 4)
sc.pl.umap(adata, color=['mclust_smoothed', 'kmeans_smoothed', 'Ground Truth'],
           title=[f'Mclust (ARI={results["mclust"][1]:.2f})',
                  f'K-means (ARI={results["kmeans"][1]:.2f})',
                  "Ground Truth"])

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, color=['mclust_smoothed', 'kmeans_smoothed', 'Ground Truth'],
              title=[f'Mclust (ARI={results["mclust"][1]:.2f})',
                     f'K-means (ARI={results["kmeans"][1]:.2f})',
                     "Ground Truth"],
              ncols=2)
