import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glasbey

# File path to the uploaded file
file_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\matrix_mutations_PTA.txt'

# Load the data
df = pd.read_csv(file_path, delimiter=r'\s+', header=0, quotechar='"')

# Select sample columns to process
# sample_columns = [f'wgs_S11340Nr{i}' for i in range(4, 8)]
sample_columns = [f'S8506Nr{i}' for i in range(3, 7)]
# Extract ALT and TOTAL from each sample column
for col in sample_columns:
    df[f'{col}_ALT'] = df[col].str.split('/').str[0].astype(float)
    df[f'{col}_Total'] = df[col].str.split('/').str[1].astype(float)

# Drop the original sample columns
df = df.drop(columns=sample_columns)

# Prepare a cleaned DataFrame with ALT and TOTAL columns
df_clean = df[[f'{col}_ALT' for col in sample_columns] + [f'{col}_Total' for col in sample_columns]].copy()

# Compute VAF for samples 2–7 (VAF = ALT/TOTAL)
# Note: samples correspond to suffixes in sample_columns
for col in sample_columns:
    alt_col = f'{col}_ALT'
    total_col = f'{col}_Total'
    vaf_col = f'{col}_VAF'
    # Avoid division by zero by replacing zeros with NaN then fill back with 0
    df_clean[vaf_col] = df_clean[alt_col] / df_clean[total_col].replace(0, pd.NA)

df_clean = df_clean.fillna(0.0)  # If there were any zero divisions, they become 0

# Only keep VAF columns for clustering/visualization
vaf_columns = [f'{col}_VAF' for col in sample_columns]
data = df_clean[vaf_columns]

# Remove rows that are all zero VAFs (optional)
data = data[(data.T != 0).any()]

# Scale data before PCA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# PCA for visualization
pca = PCA(n_components=2)
pcs = pca.fit_transform(data_scaled)
df_clean = df_clean.loc[data.index]  # Align with filtered data
df_clean['PC1'] = pcs[:, 0]
df_clean['PC2'] = pcs[:, 1]

# Define a threshold to consider a VAF "positive"
vaf_threshold = 0.01

# Define pattern order
pattern_order = [
    "1000","0100","0010","0001","1100","1010","1001",
    "0110","0101","0011","1110","1101","1011","0111","1111"
]

def assign_pattern(row):
    return "".join('1' if row[c] > vaf_threshold else '0' for c in vaf_columns)

df_clean['Pattern'] = data.apply(assign_pattern, axis=1)

# Assign clusters based on pattern
pattern_to_cluster = {p: i for i, p in enumerate(pattern_order)}
df_clean['Cluster'] = df_clean['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))

# Sort by cluster for heatmap visualization
df_clean = df_clean.sort_values('Cluster')

# Plot heatmap of VAFs sorted by cluster
plt.figure(figsize=(8, 6))
sns.heatmap(data.loc[df_clean.index])
plt.title('VAFs Sorted by Cluster')
plt.xlabel('Samples')
plt.yticks([])
plt.show()

# Prepare colors for clusters
unique_clusters = df_clean['Pattern'].unique()
cmap = glasbey.create_palette(palette_size=len(unique_clusters), as_hex=True)
color_mapping = {cluster: cmap[i] for i, cluster in enumerate(unique_clusters)}
df_clean['Color'] = df_clean['Pattern'].map(color_mapping)

# Scatter plot in PCA space colored by cluster
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['PC1'], df_clean['PC2'], c=df_clean['Color'], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters Based on VAF Barcodes')

handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_mapping[cluster], label=cluster, markersize=8)
    for cluster in unique_clusters
]
plt.legend(handles=handles, title='Cluster')
plt.show()

# Heatmap of cluster means
df_cluster_means = df_clean.groupby('Pattern')[vaf_columns].mean()
plt.figure(figsize=(8,6))
sns.heatmap(df_cluster_means, annot=True)
plt.title('Mean VAF per Cluster and Sample')
plt.xlabel('Samples')
plt.ylabel('Cluster')
plt.show()
