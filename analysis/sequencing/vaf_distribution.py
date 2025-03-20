import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

# =============================================================================
# Settings & File Loading
# =============================================================================
file_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\matrix_mutations_updates.txt'
sample_columns = [f'wgs_S11340Nr{i}' for i in range(4, 8)]

# Read the file once into a raw dataframe.
df_raw = pd.read_csv(file_path, delimiter=r'\s+', header=0, quotechar='"')

# =============================================================================
# PART A: Analysis & VAF Calculations (for VAF distribution & per‐duct charts)
# =============================================================================
df_analysis = df_raw.copy()
# Compute ALT, Total and VAF for each sample column.
for col in sample_columns:
    df_analysis[f'{col}_ALT'] = df_analysis[col].str.split('/').str[0].astype(float)
    df_analysis[f'{col}_Total'] = df_analysis[col].str.split('/').str[1].astype(float)
    df_analysis[f'{col}_VAF'] = df_analysis[f'{col}_ALT'] / df_analysis[f'{col}_Total'].replace(0, pd.NA)
df_analysis = df_analysis.fillna(0.0)
vaf_columns = [f'{col}_VAF' for col in sample_columns]

# Filter rows using only the VAF columns.
df_clean = df_analysis[vaf_columns].copy()
data = df_clean.copy()
mask = (data > 0).sum(axis=1) >= 1
data = data[mask]
df_clean = df_clean.loc[mask]
# mask = (data > 0.1).sum(axis=1) > 0
# data = data[mask]
# df_clean = df_clean.loc[mask]
# original_len = len(df_clean)

# --- Barcode, Clone & Category Assignment ---
barcode_to_clone = {}
def assign_clone_label(barcode):
    if barcode == '1111':
        return "Mother clone"
    else:
        if barcode not in barcode_to_clone:
            barcode_to_clone[barcode] = chr(65 + len(barcode_to_clone))
        return barcode_to_clone[barcode]

df_clean['barcode'] = df_clean.apply(
    lambda row: ''.join(['1' if row[col] > 0 else '0' for col in vaf_columns]), axis=1)
df_clean['clone'] = df_clean['barcode'].apply(assign_clone_label)

def assign_category(barcode):
    if barcode == "1111":
        return "fully positive"
    elif barcode.count("1") == 1:
        return "single positive"
    else:
        return "multiple positive"
df_clean['category'] = df_clean['barcode'].apply(assign_category)

# --- Global Average VAF per Clone ---
def compute_avg_vaf(group):
    vals = group[vaf_columns].values.flatten()
    vals = vals[vals > 0]
    return np.mean(vals) if vals.size > 0 else 0
avg_vaf_per_clone = df_clean.groupby('clone').apply(compute_avg_vaf)
print("Average VAF per clone:")
print(avg_vaf_per_clone)

# --- Per-Duct Raw VAF Sum ---
duct_sum_per_clone = df_clean.groupby('clone')[vaf_columns].sum()
duct_sum_by_clone = duct_sum_per_clone.T
rename_dict_analysis = {
    f'wgs_S11340Nr4_VAF': '6',
    f'wgs_S11340Nr5_VAF': '8',
    f'wgs_S11340Nr6_VAF': '11',
    f'wgs_S11340Nr7_VAF': '15'
}
duct_sum_by_clone.index = [rename_dict_analysis.get(col, col) for col in duct_sum_by_clone.index]
print("Per duct VAF sums by clone (raw sums):")
print(duct_sum_by_clone)

# --- VAF Distribution Plot by Category ---
category_vaf_values = {}
for cat in df_clean['category'].unique():
    values = df_clean.loc[df_clean['category'] == cat, vaf_columns].values.flatten()
    values = values[values > 0]
    category_vaf_values[cat] = values
category_order = ["single positive", "multiple positive", "fully positive"]
cat_vaf_list = [category_vaf_values[cat] for cat in category_order if cat in category_vaf_values]

plt.figure(figsize=(8, 5))
plt.hist(cat_vaf_list, bins=30, range=(0, 1), stacked=True,
         label=[cat for cat in category_order if cat in category_vaf_values])
plt.xlabel('VAF')
plt.ylabel('Frequency')
plt.title('VAF Distribution by Category')
plt.legend(title="Category")

# --- Per-Duct Stacked Bar Chart: Average VAF Contributions per Clone ---
rep_barcodes = df_clean.groupby('clone')['barcode'].first()
included_clones = [clone for clone in avg_vaf_per_clone.index
                   if (clone != "Mother clone") and (rep_barcodes[clone].count("1") > 1)]
print("Clones included for per-duct plot:", included_clones)

per_duct_contrib = {duct: {} for duct in vaf_columns}
for clone in included_clones:
    for duct in vaf_columns:
        per_duct_contrib[duct][clone] = avg_vaf_per_clone[clone] if duct_sum_per_clone.loc[clone, duct] > 0 else 0
per_duct_df = pd.DataFrame(per_duct_contrib).T
per_duct_df.index = [rename_dict_analysis.get(duct, duct) for duct in per_duct_df.index]
print("Individual clone contributions per duct:")
print(per_duct_df)

# --- Global Clone Color Mapping (using tab20 colormap) ---
global_clones = sorted(set(df_clean['clone'].unique()), key=lambda x: (x != "Mother clone", x))
global_clone_color_map = dict(zip(global_clones, sns.color_palette("tab20", len(global_clones))))
colors = [global_clone_color_map[clone] for clone in per_duct_df.columns]

per_duct_df.plot(kind='bar', stacked=True, color=colors, figsize=(8, 5))
plt.xlabel('Duct')
plt.ylabel('Average VAF Contribution')
plt.title('Per Duct Average VAF per Clone (Individual Contributions)')
plt.legend(title='Clone')

# =============================================================================
# PART B: Heatmap (Ordered by Cluster & Average VAF, with Consistent Clone Colors)
# =============================================================================
df_heat = df_raw.copy()
for col in sample_columns:
    df_heat[f'{col}_ALT'] = df_heat[col].str.split('/').str[0].astype(float)
    df_heat[f'{col}_Total'] = df_heat[col].str.split('/').str[1].astype(float)
df_heat = df_heat.drop(columns=sample_columns)
cols_alt_total = [f'{col}_ALT' for col in sample_columns] + [f'{col}_Total' for col in sample_columns]
df_heat_clean = df_heat[cols_alt_total].copy()
for col in sample_columns:
    alt_col = f'{col}_ALT'
    total_col = f'{col}_Total'
    vaf_col = f'{col}_VAF'
    df_heat_clean[vaf_col] = df_heat_clean[alt_col] / df_heat_clean[total_col].replace(0, pd.NA)
df_heat_clean = df_heat_clean.fillna(0.0)
vaf_columns_heat = [f'{col}_VAF' for col in sample_columns]
data_heat = df_heat_clean[vaf_columns_heat]

mask = (data_heat > 0).sum(axis=1) >= 1
data_heat = data_heat[mask]
df_heat_clean = df_heat_clean.loc[mask]
mask = (data_heat > 0.1).sum(axis=1) > 0
data_heat = data_heat[mask]
df_heat_clean = df_heat_clean.loc[mask]
original_len_heat = len(df_heat_clean)

mask2 = ((data_heat > 0).sum(axis=1) >= 2) & ((data_heat > 0).sum(axis=1) <= 3)
df_heat_clean2 = df_heat_clean.loc[mask2]
mask3 = (data_heat > 0).sum(axis=1) == 4
avg_vaf = data_heat[mask3].mean(axis=1)
print(f"Average VAF for fully positive: {avg_vaf.mean()}")
mask1 = (data_heat > 0).sum(axis=1) == 4
fraction = len(data_heat[mask1]) / original_len_heat
print(f"Fraction of rows with 4 positive VAFs: {fraction:.2%}")
mask1 = (data_heat > 0).sum(axis=1) == 1
fraction = len(data_heat[mask1]) / original_len_heat
print(f"Fraction of rows with 1 positive VAF: {fraction:.2%}")

# --- Ordering Mutations by Cluster and Average VAF ---
vaf_threshold = 0
pattern_order = []
for r in range(1, 5):
    for combo in itertools.combinations(range(4), r):
        pattern = ['0'] * 4
        for bit_index in combo:
            pattern[bit_index] = '1'
        pattern_order.append("".join(pattern))

def assign_pattern(row):
    return "".join('1' if row[c] > vaf_threshold else '0' for c in vaf_columns_heat)

df_heat_clean['Pattern'] = data_heat.apply(assign_pattern, axis=1)
pattern_to_cluster = {p: i for i, p in enumerate(pattern_order)}
df_heat_clean['Cluster'] = df_heat_clean['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))
df_heat_clean['Average_VAF'] = data_heat[data_heat > 0].mean(axis=1)
# Order by intensity then by cluster.
df_heat_clean = df_heat_clean.sort_values('Average_VAF')
df_heat_clean = df_heat_clean.sort_values('Cluster', kind="stable")

# Rename ducts for the heatmap.
rename_dict_heat = {
    'wgs_S11340Nr4_VAF': 'HM6',
    'wgs_S11340Nr5_VAF': 'HM8',
    'wgs_S11340Nr6_VAF': 'HM11',
    'wgs_S11340Nr7_VAF': 'HM15'
}
df_heat_clean = df_heat_clean.rename(columns=rename_dict_heat)
vaf_columns_heat = list(rename_dict_heat.values())
data_heat = df_heat_clean[vaf_columns_heat]

# --- Heatmap Clone Assignment (for consistent color annotation) ---
df_heat_clean['barcode'] = df_heat_clean.apply(
    lambda row: ''.join(['1' if row[c] > 0 else '0' for c in vaf_columns_heat]), axis=1)
df_heat_clean['clone'] = df_heat_clean['barcode'].apply(assign_clone_label)
# (Keep ordering by Cluster & Average_VAF.)

# Update global clone mapping with any new clones from the heatmap.
global_clones = sorted(set(global_clone_color_map.keys()).union(set(df_heat_clean['clone'].unique())),
                        key=lambda x: (x != "Mother clone", x))
global_clone_color_map = dict(zip(global_clones, sns.color_palette("tab20", len(global_clones))))

mutation_colors = df_heat_clean['clone'].map(global_clone_color_map)
# Transpose so that mutations are on the x-axis and ducts on the y-axis.
data_heat_transposed = data_heat.T.loc[:, df_heat_clean.index]

# Create a clustermap without clustering (to preserve ordering) and annotate columns.
g = sns.clustermap(data_heat_transposed,
                   col_cluster=False,
                   row_cluster=False,
                   col_colors=mutation_colors,
                   figsize=(12, 6))
g.ax_heatmap.set_xlabel('Mutations (ordered by Cluster & intensity)')
g.ax_heatmap.set_ylabel('Duct')
# =============================================================================
# PART C: VAF Distribution Histogram per Clone (Stacked Histogram)
# =============================================================================
# Use df_clean and restrict to clones included in the per-duct plot.
df_vaf_clone = df_clean[df_clean['clone'].isin(included_clones)].copy()
# Reshape to long format.
df_vaf_long = df_vaf_clone.melt(id_vars=['clone'], value_vars=vaf_columns,
                                var_name='Duct', value_name='VAF')
df_vaf_long = df_vaf_long[df_vaf_long['VAF'] > 0]

# Create a list of VAF arrays for each clone.
clone_data = [df_vaf_long[df_vaf_long['clone'] == clone]['VAF'].values
              for clone in included_clones]
# Get the colors for each clone using the global color mapping.
clone_colors = [global_clone_color_map[clone] for clone in included_clones]

plt.figure(figsize=(8, 6))
plt.hist(clone_data, bins=30, range=(0, 1), stacked=True, color=clone_colors,
         label=included_clones)
plt.xlabel('VAF')
plt.ylabel('Frequency')
plt.title('VAF Distribution per Clone (Stacked Histogram)')
plt.legend(title='Clone')

# =============================================================================
# PART D: Heatmap for Clones with Multiple (but not all) Positive VAFs
# =============================================================================
# Compute category for heatmap data (if not already computed)
df_heat_clean['category'] = df_heat_clean['barcode'].apply(assign_category)

# Filter to only include rows with "multiple positive" (i.e. not fully or single positive).
df_heat_multi = df_heat_clean[df_heat_clean['category'] == "multiple positive"].copy()

# Instead of reassigning clone labels, use the existing clone column.
# Determine the subset of clones present in this zoomed in heatmap.
subset_clones = sorted(set(df_heat_multi['clone'].unique()),
                         key=lambda x: (x != "Mother clone", x))
# Restrict the global color mapping to only those clones.
multi_clone_color_map = {clone: global_clone_color_map[clone] for clone in subset_clones if clone in global_clone_color_map}

# Map the colors for each mutation row in the subset.
mutation_colors_multi = df_heat_multi['clone'].map(multi_clone_color_map)

# Transpose so that mutations are on the x-axis and ducts on the y-axis.
data_heat_transposed_multi = data_heat.T.loc[:, df_heat_multi.index]

# Create the clustermap without clustering (to preserve ordering) and annotate columns.
g_multi = sns.clustermap(data_heat_transposed_multi,
                         col_cluster=False,
                         row_cluster=False,
                         col_colors=mutation_colors_multi,
                         figsize=(12, 6))
g_multi.ax_heatmap.set_xlabel('Mutations (ordered by Cluster & intensity)')
g_multi.ax_heatmap.set_ylabel('Duct')
plt.title('Heatmap: Clones with Multiple Positive VAFs')

# =============================================================================
# PART E: Correlation Analysis & Mutation Pattern Grouping for Clones with Multiple Positive Ducts
# =============================================================================

# --- A. Correlation Heatmaps for Each Clone ---
# We'll compute the correlation of VAFs (for ducts) only for mutations with "multiple positive" pattern.
clones_to_analyze = included_clones  # clones with >1 positive duct (excluding Mother clone)
num_clones = len(clones_to_analyze)

fig, axes = plt.subplots(nrows=1, ncols=num_clones, figsize=(5*num_clones, 4), squeeze=False)
axes = axes.flatten()

for ax, clone in zip(axes, clones_to_analyze):
    # Filter for rows in the clone with multiple positive ducts.
    subset = df_clean[(df_clean['clone'] == clone) & (df_clean['category'] == "multiple positive")]
    if subset.shape[0] < 2:
        ax.text(0.5, 0.5, f"Not enough data for Clone {clone}",
                horizontalalignment='center', verticalalignment='center')
        ax.set_title(f"Clone {clone}")
        ax.axis('off')
    else:
        # Compute correlation matrix between VAF columns.
        corr = subset[vaf_columns].corr()
        # Rename indices and columns using the rename dictionary for clarity.
        corr = corr.rename(index=rename_dict_analysis, columns=rename_dict_analysis)
        sns.heatmap(corr, annot=True, ax=ax, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_title(f"Correlation of VAFs for Clone {clone}")

plt.tight_layout()

# =============================================================================
# PART F: Violin Plot of VAF Distribution per Clone
# =============================================================================
# Using the same long-format data from PART C.
plt.figure(figsize=(8, 6))
sns.violinplot(x='clone', y='VAF', data=df_vaf_long, order=included_clones,
               palette=global_clone_color_map)
plt.xlabel('Clone')
plt.ylabel('VAF')
plt.title('Violin Plot of VAF Distribution per Clone')
plt.show()


