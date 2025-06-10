import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pathlib import Path
file_path = Path(r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\mutation_matrix_updated_full.txt')

# first line has only the sample names; use it to build the column list
with file_path.open() as fh:
    header_line = fh.readline().strip()
sample_names = [s.strip('"') for s in header_line.split()]
cols = ['Mutation'] + sample_names                           # 7 columns total

df = pd.read_csv(
    file_path,
    sep=r'\s+',
    engine='python',
    quoting=3,
    names=cols,
    skiprows=1
)


orig_samples = ['HM6', 'HM8', 'HM11', 'HM15']
sample_columns = orig_samples[-1:] + orig_samples[:-1]        # cyclic right‑shift
data = df[sample_columns].astype(float)                       # VAF matrix (N×4)


mask_nonzero = (data > 0.1).any(axis=1)
df = df.loc[mask_nonzero]
data = data.loc[df.index]
original_len = len(df)

# quick metrics / histograms
fully_pos = (data > 0).sum(axis=1) == 4
avg_vaf   = data[fully_pos].mean(axis=1)
print(f'Average VAF for fully positive: {avg_vaf.mean():.4f}')

print(f"Total number of mutations: {len(df)}")

plt.figure(figsize=(6,4))
plt.hist(data[fully_pos].to_numpy().ravel(), bins=15, range=(0,1))
plt.axvline(avg_vaf.mean(), color='r', ls='--')
plt.legend([f'avg {avg_vaf.mean():.3f}'])
plt.xlabel('VAF')
plt.ylabel('Frequency')
plt.title('VAFs for fully positive mutations')

# fractions (1‑, 2‑3‑, 4‑positive rows)
for k,label in [(1,'1'),(4,'4'),('2‑3','2‑3')]:
    if k=='2‑3':
        m = ((data>0).sum(axis=1).between(2,3))
    else:
        m = ((data>0).sum(axis=1)==k)
    print(f'Fraction with {label} positive VAFs: {m.mean():.2%}')

# binary pattern
vaf_threshold = 0
def assign_pattern(row):
    return ''.join('1' if v>vaf_threshold else '0' for v in row)

df['Pattern'] = data.apply(assign_pattern, axis=1)

# remove duplicates
pattern_order = [''.join('1' if i in c else '0' for i in range(4))
                 for r in range(1,5)
                 for c in itertools.combinations(range(4), r)]
excluded = {'0111'}            # keep your previous exclusions
sel = df['Pattern'].str.count('1').between(2,3) & ~df['Pattern'].isin(excluded)
print(f'Fraction of barcodes with >1&<4 positives (excl. {excluded}): {sel.mean():.2%}')

pattern_to_cluster = {p:i for i,p in enumerate(pattern_order)}
df['Cluster'] = df['Pattern'].map(pattern_to_cluster).fillna(-1).astype(int)

df['Average_VAF'] = data.where(data>0).mean(axis=1)
df = df.sort_values(['Cluster','Average_VAF'])

plt.figure(figsize=(15,3))
sns.heatmap(df[sample_columns].T, cbar_kws=dict(label='VAF'))
plt.xlabel('Mutations')
plt.xticks([])
out_png = file_path.with_name('heatmap_vaf_batch2_4_samples.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')


# === Annotate with barcode & biological type ===============================
df['Barcode'] = df['Pattern']

pattern_to_type = {
    # striping type 1
    '1011': 'striping type 1', '1010': 'striping type 1', '1001': 'striping type 1',
    # striping type 2
    '1100': 'striping type 2', '1101': 'striping type 2', '1110': 'striping type 2',
    '0110': 'striping type 2', '0101': 'striping type 2', '0011': 'striping type 2',
    # non-striping
    '0111': 'nonstriping',
    # adult
    '1000': 'adult', '0100': 'adult', '0010': 'adult', '0001': 'adult',
    # embryonic
    '1111': 'embryonic'
}
df['Type'] = df['Barcode'].map(pattern_to_type).fillna('other')

# === Save annotated mutation matrix ========================================
cols_out = ['Mutation', *sample_columns, 'Barcode', 'Type', 'Average_VAF']
out_txt = file_path.with_name('mutation_matrix_annotated.txt')
df[cols_out].to_csv(out_txt, sep='\t', index=False, quoting=3)
print(f'Annotated matrix saved to: {out_txt}')

plt.show()

print(f'Heat‑map saved to: {out_png}')
