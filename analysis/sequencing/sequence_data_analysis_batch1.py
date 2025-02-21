import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# File path to the uploaded file
file_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\matrix_mutations_PTA.txt'

# Load the data
df = pd.read_csv(file_path, delimiter=r'\s+', header=0, quotechar='"')

# Select sample columns to process
sample_columns = [f'S8506Nr{i}' for i in range(3, 7)]

# Extract ALT and TOTAL from each sample column
for col in sample_columns:
    df[f'{col}_ALT'] = df[col].str.split('/').str[0].astype(float)
    df[f'{col}_Total'] = df[col].str.split('/').str[1].astype(float)

# Drop the original sample columns
df = df.drop(columns=sample_columns)

# Prepare a cleaned DataFrame with ALT and TOTAL columns
df_clean = df[[f'{col}_ALT' for col in sample_columns] + [f'{col}_Total' for col in sample_columns]].copy()

# Compute VAF for samples
for col in sample_columns:
    alt_col = f'{col}_ALT'
    total_col = f'{col}_Total'
    vaf_col = f'{col}_VAF'
    df_clean[vaf_col] = df_clean[alt_col] / df_clean[total_col].replace(0, pd.NA)

# delete rows with nan
df_clean = df_clean.dropna()

# Only keep VAF columns for clustering/visualization
vaf_columns = [f'{col}_VAF' for col in sample_columns]
data = df_clean[vaf_columns]


# Scale data before PCA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Define a threshold to consider a VAF "positive"
vaf_threshold = 0.2
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

#
# # Rename columns before creating data
# rename_dict = {
#     'wgs_S11340Nr4_VAF': 'Annotation 6',
#     'wgs_S11340Nr5_VAF': 'Annotation 8',
#     'wgs_S11340Nr6_VAF': 'Annotation 11',
#     'wgs_S11340Nr7_VAF': 'Annotation 15'
# # }
# df_clean = df_clean.rename(columns=rename_dict)
#
# # Now define vaf_columns based on renamed columns
# vaf_columns = list(rename_dict.values())  # ['Ann 6 (WGS nr4)', 'Ann 8 (WGS nr5)', 'Ann 11 (WGS nr6)', 'Ann 15 (WGS nr7)']
data = df_clean[vaf_columns]

# set dtype to float
data = data.astype(float)


# For the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.loc[df_clean.index])
plt.xlabel('Samples')
plt.yticks([])
plt.show()