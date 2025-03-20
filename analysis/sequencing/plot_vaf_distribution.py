import pandas as pd
import matplotlib.pyplot as plt

file_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\matrix_mutations_updates.txt'
sample_columns = [f'wgs_S11340Nr{i}' for i in range(4, 8)] # only last 4 samples

# Read the file once into a raw dataframe.
df_raw = pd.read_csv(file_path, delimiter=r'\s+', header=0, quotechar='"')

df_analysis = df_raw.copy()
# Compute ALT, Total and VAF for each sample column.
for col in sample_columns:
    df_analysis[f'{col}_ALT'] = df_analysis[col].str.split('/').str[0].astype(float)
    df_analysis[f'{col}_Total'] = df_analysis[col].str.split('/').str[1].astype(float)
    df_analysis[f'{col}_VAF'] = df_analysis[f'{col}_ALT'] / df_analysis[f'{col}_Total'].replace(0, pd.NA)
df_analysis = df_analysis.fillna(0.0)
vaf_columns = [f'{col}_VAF' for col in sample_columns]

# Filter rows to make sure they have at least 1 positive VAF.
df_clean = df_analysis[vaf_columns].copy()
# data = df_clean.copy()
# mask = (data > 0).sum(axis=1) >= 1
# data = data[mask]
# df_clean = df_clean.loc[mask]

# Barcoding based on positive/negative VAFs
df_clean['barcode'] = df_clean.apply(
    lambda row: ''.join(['1' if row[col] > 0 else '0' for col in vaf_columns]), axis=1)


# Assign barcodes to categories
def assign_category(barcode):
    if barcode == "1111":
        return "fully positive"
    elif barcode.count("1") == 1:
        return "single positive"
    else:
        return "multiple positive"


df_clean['category'] = df_clean['barcode'].apply(assign_category)

# Plotting histogram
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
plt.title('VAF Distribution')
plt.legend()
plt.show()
