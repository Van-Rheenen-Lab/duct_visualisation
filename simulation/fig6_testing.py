import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1. Load & Aggregate Puberty CSV Data
# --------------------------------------------------------------------
puberty_folder = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\FIGURE DATA\lengths CNV files puberty"
all_sums = []  # Will hold per-file clone (color) sums

for filename in os.listdir(puberty_folder):
    if filename.endswith(".csv"):
        csv_path = os.path.join(puberty_folder, filename)
        # Read CSV with ";" delimiter
        df = pd.read_csv(csv_path, delimiter=";")
        # Rename "Length µm" or "Length Âµm" to a consistent "Length_um"
        df.rename(columns={
            "Length µm": "Length_um",
            "Length Âµm": "Length_um"
        }, inplace=True)
        # Group by Image and Class (each clone/color) and sum the lengths per clone
        group_sums = df.groupby(["Image", "Class"])["Length_um"].sum().reset_index()
        all_sums.append(group_sums)

# Combine results from all files
puberty_data = pd.concat(all_sums, ignore_index=True)
# Average over all clones (i.e. each group of [Image, Class]) to get one mean value for puberty
puberty_mean = puberty_data.groupby(["Image", "Class"])["Length_um"].sum().mean()

# --------------------------------------------------------------------
# 2. Load Adult Data
# --------------------------------------------------------------------
# Adult 255 file (semicolon separated)
adult_255_file = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\FIGURE DATA\lengths adult\adult_pooled_255.csv"
adult_255_df = pd.read_csv(adult_255_file, header=None, delimiter=";")

# ignore nans
adult_255_mean = adult_255_df.dropna().values.flatten().mean()


# Adult 64 file (comma separated)
adult_64_file = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\FIGURE DATA\lengths adult\measurements length clones adult b1p 64.csv"
adult_64_df = pd.read_csv(adult_64_file, delimiter=',')
# Here we use the last column (adjust if needed) to compute the mean length
adult_64_mean = adult_64_df[adult_64_df.columns[-1]].values.flatten().mean()

# --------------------------------------------------------------------
# 3. Plot: One Plot with Time on X and Average Length on Y
# --------------------------------------------------------------------
# Define time points for each condition.
# We'll assume:
#   - Adult 64 is at day 64.
#   - Puberty is at day 255.
#   - Adult 255 is at day 255.
# To avoid overlap at day 255, we can add a slight offset.
time_adult = [64, 255]    # Adult 255 is shifted a bit (255.2) for clarity.
avg_adult = [adult_64_mean, adult_255_mean]
time_puberty = [255]        # Puberty remains at 255.
avg_puberty = [puberty_mean]

plt.figure(figsize=(8,6))
# Plot Adult data (blue circles)
plt.scatter(time_adult, avg_adult, color='blue', label='Adult', s=100)
# Plot Puberty data (red squares)
plt.scatter(time_puberty, avg_puberty, color='red', marker='s', label='Puberty', s=100)
# log scale for better visualization
plt.yscale("log")
# set bottom limit to 1
plt.ylim(bottom=1)


plt.xlabel("Time (days)")
plt.ylabel("Average Length (µm)")
plt.title("Average Length vs Time")
plt.legend()
plt.grid(True)
plt.show()
