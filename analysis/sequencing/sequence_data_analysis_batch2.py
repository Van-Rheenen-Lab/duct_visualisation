import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

file_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\matrix_mutations_updates.txt'

# Load the data
df = pd.read_csv(file_path, delimiter=r'\s+', header=0, quotechar='"')

# Select sample columns to process
sample_columns = [f'wgs_S11340Nr{i}' for i in range(2, 8)]

# Extract ALT and TOTAL from each sample column
for col in sample_columns:
    df[f'{col}_ALT'] = df[col].str.split('/').str[0].astype(float)
    df[f'{col}_Total'] = df[col].str.split('/').str[1].astype(float)

# Drop the original sample columns
df = df.drop(columns=sample_columns)

# Prepare a cleaned DataFrame with ALT and TOTAL columns
df_clean = df[
    [f'{col}_ALT' for col in sample_columns] +
    [f'{col}_Total' for col in sample_columns]
].copy()

# Compute VAF for samples
for col in sample_columns:
    alt_col = f'{col}_ALT'
    total_col = f'{col}_Total'
    vaf_col = f'{col}_VAF'
    df_clean[vaf_col] = df_clean[alt_col] / df_clean[total_col].replace(0, pd.NA)

df_clean = df_clean.fillna(0.0)

# Only keep VAF columns for clustering/visualization
vaf_columns = [f'{col}_VAF' for col in sample_columns]
data = df_clean[vaf_columns]

original_len = len(df_clean)

# Remove rows that do not have at least 2 columns with VAF > 0
mask = (data > 0).sum(axis=1) >= 1
data = data[mask]
df_clean = df_clean.loc[mask]

# Calculate the fraction of rows that have >= 2 and <= 5 positive VAFs
mask2 = ((data > 0).sum(axis=1) >= 2) & ((data > 0).sum(axis=1) <= 5)
df_clean2 = df_clean.loc[mask2]

# calculate average VAF for fully positive rows
mask3 = (data > 0).sum(axis=1) == 6
avg_vaf = data[mask3].mean(axis=1)
print(f"Average VAF for fully positive: {avg_vaf.mean()}")
# plot histogram of average VAF for fully positive rows from 0 to 1
vals_flat = data[mask3].to_numpy().flatten()
plt.hist(vals_flat, bins=15, range=(0, 1))
# draw average VAF line
plt.axvline(avg_vaf.mean(), color='r', linestyle='dashed', linewidth=1)
# make legend
plt.legend([f'Average VAF: {avg_vaf.mean():.4}'])
plt.xlabel('VAF')
plt.ylabel('Frequency')
plt.title('VAFs for fully positive mutations')


fraction = len(df_clean2) / original_len
print(f"Fraction of rows with at least 2 and max 5 positive VAFs: {fraction:.2%}")

# calculate fraction of rows with 6 positive VAFs
mask1 = (data > 0).sum(axis=1) == 6
fraction = len(data[mask1]) / original_len
print(f"Fraction of rows with 4 positive VAFs: {fraction:.2%}")

# calculate fraction of rows with 1 positive VAF
mask1 = (data > 0).sum(axis=1) == 1
fraction = len(data[mask1]) / original_len
print(f"Fraction of rows with 1 positive VAF: {fraction:.2%}")

# Define a threshold to consider a VAF "positive"
vaf_threshold = 0

# Generate an ordered pattern list, from 1-positive-bit to 6-positive-bits
pattern_order = []
for r in range(1, 7):
    for combo in itertools.combinations(range(6), r):
        pattern = ['0'] * 6
        for bit_index in combo:
            pattern[bit_index] = '1'
        pattern_order.append("".join(pattern))

def assign_pattern(row):
    return "".join('1' if row[c] > vaf_threshold else '0' for c in vaf_columns)

df_clean['Pattern'] = data.apply(assign_pattern, axis=1)

# Assign clusters based on pattern
pattern_to_cluster = {p: i for i, p in enumerate(pattern_order)}
df_clean['Cluster'] = df_clean['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))

# Sort by cluster for heatmap visualization
df_clean = df_clean.sort_values('Cluster')

# Rename columns for clarity
rename_dict = {
    'wgs_S11340Nr2_VAF': 'HM1',
    'wgs_S11340Nr3_VAF': 'HM3',
    'wgs_S11340Nr4_VAF': 'HM6',
    'wgs_S11340Nr5_VAF': 'HM8',
    'wgs_S11340Nr6_VAF': 'HM11',
    'wgs_S11340Nr7_VAF': 'HM15'
}
df_clean = df_clean.rename(columns=rename_dict)

vaf_columns = list(rename_dict.values())
data = df_clean[vaf_columns]

plt.figure(figsize=(8, 6))
sns.heatmap(data)
plt.ylabel('Mutations')
plt.yticks([])
plt.show()
