######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import scanpy as sc
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import seaborn as sb

######################
# Page Title
######################

# image = Image.open('nsclc-logo.jpeg')

# st.image(image, use_column_width=True)

st.write("""
# Non-small cell lung cancer Web App

This app visualize the scRNA-seq data of Non-small cell lung cancer!

Data obtained from the CancerSEM database.
""")


######################
# Input scRNA-seq data (Side Panel)
######################

st.sidebar.header('User scRNA-seq data (please upload files)')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file with scRNA-seq data", type="tsv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file,sep='\t')
    st.write(data)

use_example_file = st.sidebar.checkbox(
    "Use example file", False, key='1', help="Use in-built example file to demo the app"
)

# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block
if use_example_file:
    uploaded_file = './Data/LUAD-003-01-1A.gene.expression.matrix.tsv'
    data = pd.read_csv(uploaded_file,sep='\t')
    # st.write(data)

st.sidebar.header('User differential expressed gene data (please upload files)')

deg_file = st.sidebar.file_uploader("Choose a CSV file with scRNA-seq DEG data", type="csv")
if deg_file is not None:
    deg_df = pd.read_csv(deg_file)
    st.write(deg_df)

use_example_file = st.sidebar.checkbox(
    "Use example file", False, key='2', help="Use in-built example file to demo the app"
)

# If CSV is not uploaded and checkbox is filled, use values from the example file
# and pass them down to the next if block
if use_example_file:
    deg_file = './Data/LUAD-003-01-1A.DEGs.for.all.celltypes.tsv'
    deg_df = pd.read_csv(deg_file,sep='\t')
    # st.write(data)


st.header('Read data using scanpy')

#Load data
ge_matrix = uploaded_file
adata = sc.read(ge_matrix, cache=True)
adata = adata.transpose()
cell_count = adata.X.shape[0]
gene_count = adata.X.shape[1]
st.write('Number of cells in the data: ', cell_count)
st.write('Number of genes in the data: ', gene_count)

gene_id = pd.DataFrame(columns=['gene'])
gene_id['gene'] = list(data.index)
gene_id.head()
adata.var = gene_id

st.header('Dimension reduction visualization')
from matplotlib.pyplot import rc_context
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader('Visualization by PCA')
sc.pp.pca(adata, n_comps=50, use_highly_variable=False, svd_solver='arpack')
with rc_context({'figure.figsize': (10, 10)}):
    sc.pl.pca_scatter(adata)
    st.pyplot()

st.subheader('Visualization by UMAP')
# t-SNE
tsne_n_pcs = 20 # Number of principal components to use for t-SNE

# k-means
k = 18 # Number of clusters for k-means

# KNN
n_neighbors = 15 # Number of nearest neighbors for KNN graph
knn_n_pcs = 50 # Number of principal components to use for finding nearest neighbors

# UMAP
umap_min_dist = 0.3 
umap_spread = 1.0
# KNN graph
sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=knn_n_pcs)
# UMAP
sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)
# Louvain clustering
st.write('UMAP visualization with Louvain clustering')
sc.tl.louvain(adata)
with rc_context({'figure.figsize': (10, 10)}):
# Plot
    sc.pl.umap(adata, color=["louvain"],legend_loc='on data', legend_fontsize=10)
    st.pyplot()

# Leiden clustering
st.write('UMAP visualization with Leiden clustering')
sc.tl.leiden(adata)
# Plot
with rc_context({'figure.figsize': (10, 10)}):
    sc.pl.umap(adata, color=["leiden"],legend_loc='on data', legend_fontsize=10)
    st.pyplot()

st.subheader('Visualization by t-SNE')
sc.tl.tsne(adata, n_pcs=tsne_n_pcs)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=18, random_state=0).fit(adata.obsm['X_pca'])
adata.obs['kmeans'] = kmeans.labels_.astype(str)
with rc_context({'figure.figsize': (10, 10)}):
    sc.pl.tsne(adata, color=["kmeans"])
    st.pyplot()

st.header('Cell type annotation')

clusters = list(deg_df['cluster'].unique())
st.write(clusters)

st.subheader('Visualization of marker genes of CD4+ Tcm cell by UMAP')
CD4_Tcm = deg_df[deg_df['cluster']=='CD4+ Tcm'].sort_values(by=['p_val_adj','avg_logFC'])
CD4_Tcm.head()
gene_list = ['SELL','LTB','LAIR2']
adata.var_names = gene_id['gene'].tolist()
# Plot
with rc_context({'figure.figsize': (8, 8)}):
    fig = sc.pl.umap(adata, color=gene_list, color_map='viridis', legend_fontsize=18)
    fn = "./CD4+.png"
    plt.savefig(fn)
    st.pyplot(fig)
    with open(fn, "rb") as file:
     btn = st.download_button(
             label="Export image",
             data=file,
             file_name=fn,
             mime="image/png"
           )

umap_cell_cluster_dict= { 'CD4+ Tcm':['1'],
 'CD4+ Tem':['2'],
 'CD8+ effector T cells':['3'],
 'CD8+ naive T cells':['4'],
 'CD8+ Tcm':['5'],
 'CD8+ Tem':['6'],
 'Dendritic cells':['7'],
 'Endothelial cells':['8'],
 'Fibroblasts':['9'],
 'Macrophages':['9'],
 'Malignant cells':['11'],
 'Mast cells':['12'],
 'Monocytes':['13'],
 'NK cells':['14'],
 'Tregs':['15']}

# Initialize empty column in cell metadata
adata.obs['cell_type'] = np.nan

# Generate new assignments
for i in umap_cell_cluster_dict.keys():
    ind = pd.Series(adata.obs.leiden).isin(umap_cell_cluster_dict[i])
    adata.obs.loc[ind,'cell_type'] = i
with plt.rc_context({'figure.figsize': (10, 10)}):
    sc.pl.umap(adata, color=['leiden'], legend_loc='on data', legend_fontsize=12, save='umap_leiden')
    sc.pl.umap(adata, color=['cell_type'], legend_loc='on data', legend_fontsize=12, save='umap_leiden_cell_type')
    st.pyplot()
    st.pyplot()
