import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import io, filters
from shapely.geometry import shape
from rasterio.features import rasterize
from scipy.spatial import cKDTree
import networkx as nx
from scipy.interpolate import interp1d

# Define file paths with your correct paths
ecad_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
annotated_tracks_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'

# Step 1: Load the raw Ecad channel image (tif image)
ecad_image = io.imread(ecad_image_path)

# Ensure the image is 2D
if ecad_image.ndim > 2:
    ecad_image = ecad_image[:, :, 0]  # Adjust index if necessary

# Step 2: Load the duct border annotations
with open(duct_borders_path, 'r') as f:
    duct_borders = json.load(f)

# Create a mask from the duct borders
shapes = []
for feature in duct_borders['features']:
    geometry = feature['geometry']
    shapely_geom = shape(geometry)
    shapes.append((shapely_geom, 1))  # Assign a value of 1 inside the polygon

# Rasterize the shapes onto the mask
mask = rasterize(
    shapes,
    out_shape=ecad_image.shape[:2],
    fill=0,
    dtype=np.uint8,
    all_touched=False
)

# Convert mask to boolean
mask = mask.astype(bool)

# Apply the mask to the ecad_image
masked_image = ecad_image.copy()
masked_image[~mask] = 0

# Step 5: Threshold the image to detect positive signal
threshold_value = filters.threshold_otsu(masked_image[mask])  # Consider only masked pixels
binary_image = masked_image > threshold_value

# Step 6: Load the annotated tracks (duct segments and branch points)
with open(annotated_tracks_path, 'r') as f:
    annotated_tracks = json.load(f)

segments = annotated_tracks['duct_systems'][0]['segments']
branch_points = annotated_tracks['duct_systems'][0]['branch_points']

# Build the duct graph (tree) using NetworkX
G = nx.Graph()
for bp_name in branch_points.keys():
    G.add_node(bp_name, pos=(branch_points[bp_name]['x'], branch_points[bp_name]['y']))

for segment_name, segment_data in segments.items():
    start_bp = segment_data['start_bp']
    end_bp = segment_data['end_bp']
    # Calculate Euclidean distance between branch points as edge weight
    x1, y1 = branch_points[start_bp]['x'], branch_points[start_bp]['y']
    x2, y2 = branch_points[end_bp]['x'], branch_points[end_bp]['y']
    distance = np.hypot(x2 - x1, y2 - y1)
    G.add_edge(start_bp, end_bp, weight=distance)

# Choose a root node (e.g., the node with the highest betweenness centrality)
centrality = nx.betweenness_centrality(G)
root_node = max(centrality, key=centrality.get)

# Compute shortest paths and distances from root node to all other nodes
lengths = nx.single_source_dijkstra_path_length(G, root_node, weight='weight')

# Build duct points and cumulative distances
duct_points = []
duct_distances = []

for u, v, data in G.edges(data=True):
    # Get the positions of the nodes
    x1, y1 = branch_points[u]['x'], branch_points[u]['y']
    x2, y2 = branch_points[v]['x'], branch_points[v]['y']

    # Get cumulative distances to the nodes
    dist_u = lengths[u]
    dist_v = lengths[v]

    # Interpolate points along the edge
    num_points = 10  # Adjust as needed
    x_coords = np.linspace(x1, x2, num=num_points)
    y_coords = np.linspace(y1, y2, num=num_points)
    cumulative_distances = np.linspace(dist_u, dist_v, num=num_points)

    for x, y, cum_dist in zip(x_coords, y_coords, cumulative_distances):
        duct_points.append([y, x])  # Note [y, x]
        duct_distances.append(cum_dist)

# Build a KDTree of duct points
duct_tree = cKDTree(duct_points)

# For each positive pixel, find the nearest duct point
positive_pixels = np.argwhere(binary_image)

# Query the KDTree
distances_to_duct, indices = duct_tree.query(positive_pixels)

# Set a maximum distance threshold
max_distance = 10  # Adjust as needed
valid_indices = distances_to_duct <= max_distance

# Filter valid pixels
positive_pixels = positive_pixels[valid_indices]
indices = indices[valid_indices]

# Get the cumulative distances for the positive pixels
pixel_cumulative_distances = np.array(duct_distances)[indices]

# Normalize the cumulative distances
max_cumulative_distance = max(lengths.values())  # Maximum distance in the duct system
normalized_distances = pixel_cumulative_distances / max_cumulative_distance

# Get the intensities of the positive pixels
positive_intensities = masked_image[positive_pixels[:, 0], positive_pixels[:, 1]]

# Assign intensities to quartiles based on normalized distances
quartile_edges = [0, 0.25, 0.5, 0.75, 1.0]
quartile_intensities = {i: [] for i in range(1, 5)}  # Quartiles 1 to 4

for i in range(len(positive_pixels)):
    norm_dist = normalized_distances[i]
    intensity = positive_intensities[i]
    for q in range(1, 5):
        if quartile_edges[q - 1] <= norm_dist < quartile_edges[q]:
            quartile_intensities[q].append(intensity)
            break

# Plot histograms of the intensities for each quartile in subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, q in enumerate(range(1, 5)):
    intensities = quartile_intensities[q]
    if intensities:
        axs[i].hist(intensities, bins=30, edgecolor='black')
        axs[i].set_xlabel('Signal Intensity')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Quartile {q}: {int(quartile_edges[q - 1]*100)}%-{int(quartile_edges[q]*100)}% Along Ducts')
        axs[i].grid(True)
    else:
        axs[i].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axs[i].set_title(f'Quartile {q}: No Data')
        axs[i].axis('off')

plt.tight_layout()
plt.show()

# Plot the average intensity profile across all pixels
# Define common distance points
num_bins = 100  # Adjust for desired resolution
bins = np.linspace(0, 1, num_bins + 1)
bin_indices = np.digitize(normalized_distances, bins)

# Compute average intensity in each bin
binned_distances = []
binned_intensities = []
binned_std = []

for i in range(1, len(bins)):
    mask = bin_indices == i
    if np.any(mask):
        avg_distance = normalized_distances[mask].mean()
        avg_intensity = positive_intensities[mask].mean()
        std_intensity = positive_intensities[mask].std()
        binned_distances.append(avg_distance)
        binned_intensities.append(avg_intensity)
        binned_std.append(std_intensity)

# Plot the average intensity profile with standard deviation shading
plt.figure(figsize=(10, 6))
plt.plot(binned_distances, binned_intensities, color='blue', label='Mean Intensity')
plt.fill_between(binned_distances, np.array(binned_intensities) - np.array(binned_std),
                 np.array(binned_intensities) + np.array(binned_std),
                 color='blue', alpha=0.3, label='±1 Std Dev')

plt.xlabel('Normalized Distance from Origin')
plt.ylabel('Average Signal Intensity')
plt.title('Average Intensity Profile Across Ducts')
plt.legend()
plt.grid(True)
plt.show()
