import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import io, filters, morphology
from shapely.geometry import shape, LineString
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import networkx as nx
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches

# --- File Paths ---
ecad_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
annotated_tracks_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'

# --- Parameters ---
max_distance = 25
override_threshold = 400  # Set to None for automatic thresholding
step_size = 10  # Fixed step size in pixels

# --- Step 1: Load the Raw Ecad Image ---
print("Loading Ecad image...")
try:
    ecad_image = io.imread(ecad_image_path)
    print(f"Ecad image shape: {ecad_image.shape}")
except Exception as e:
    raise FileNotFoundError(f"Error loading Ecad image: {e}")

# --- Step 2: Load the Duct Border Annotations ---
print("Loading duct border annotations...")
try:
    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)
    print(f"Loaded {len(duct_borders.get('features', []))} duct border features.")
except Exception as e:
    raise FileNotFoundError(f"Error loading duct borders: {e}")

# --- Step 3: Create a Mask Based on Duct Borders ---
print("Rasterizing duct borders to create mask...")
shapes = [(shape(feature['geometry']), 1) for feature in duct_borders.get('features', [])]

mask = rasterize(
    shapes,
    out_shape=ecad_image.shape,
    fill=0,
    dtype=np.uint8,
    all_touched=False
).astype(bool)
print(f"Mask created with shape: {mask.shape}")

# --- Step 4: Apply the Mask to the Ecad Image ---
print("Applying mask to Ecad image...")
masked_image = ecad_image.copy()
masked_image[~mask] = 0
print("Mask applied.")

# --- Step 5: Threshold the Image to Detect Positive Signal ---
if override_threshold is None:
    print("Calculating Otsu's threshold...")
    if np.sum(mask) == 0:
        raise ValueError("Mask is empty. No pixels to threshold.")
    threshold_value = filters.threshold_otsu(ecad_image[mask])
    print(f"Threshold value (Otsu): {threshold_value}")
else:
    threshold_value = override_threshold
    print(f"Threshold value (Override): {threshold_value}")

binary_image = masked_image > threshold_value
print(f"Binarization complete. Number of positive pixels: {np.sum(binary_image)}")

# --- Step 6: Load the Annotated Tracks ---
print("Loading annotated tracks...")
try:
    with open(annotated_tracks_path, 'r') as f:
        annotated_tracks = json.load(f)
    print(f"Loaded {len(annotated_tracks.get('duct_systems', []))} duct systems.")
except Exception as e:
    raise FileNotFoundError(f"Error loading annotated tracks: {e}")

# Verify the presence of duct systems
if not annotated_tracks.get('duct_systems'):
    raise ValueError("No duct systems found in annotated_tracks.json.")

system = annotated_tracks['duct_systems'][0]
branch_points = system.get('branch_points', {})
segments = system.get('segments', {})

print(f"Number of branch points: {len(branch_points)}")
print(f"Number of segments: {len(segments)}")

if not segments:
    raise ValueError("No segments found in the first duct system.")

# --- Step 7: Align the Coordinate System if Necessary ---
print("Aligning coordinate system...")
# Find minimum x and y values
min_x = min([bp['x'] for bp in branch_points.values()]) if branch_points else 0
min_y = min([bp['y'] for bp in branch_points.values()]) if branch_points else 0

for segment_data in segments.values():
    for point in segment_data.get('internal_points', []):
        min_x = min(min_x, point['x'])
        min_y = min(min_y, point['y'])

print(f"Minimum x: {min_x}, Minimum y: {min_y}")

# --- Step 8: Build a Spatial Index (KDTree) for the Segments ---
print("Building spatial index (KDTree) for segments...")
segment_points = []
segment_labels = []
segment_point_indices = []  # To keep track of point indices
segment_point_coords = {}  # To store coordinates for each point along segments

point_index = 0
segment_name_to_point_indices = defaultdict(list)  # Precompute mapping

for segment_name, segment_data in segments.items():
    start_bp_name = segment_data.get('start_bp')
    end_bp_name = segment_data.get('end_bp')

    start_bp = branch_points.get(start_bp_name)
    end_bp = branch_points.get(end_bp_name)

    if start_bp is None or end_bp is None:
        print(f"Warning: Start or end branch point missing for segment {segment_name}. Skipping.")
        continue

    points = [(start_bp['x'], start_bp['y'])]

    # Add internal points if any
    internal_points = segment_data.get('internal_points', [])
    for point in internal_points:
        points.append((point['x'], point['y']))

    points.append((end_bp['x'], end_bp['y']))
    line = LineString(points)

    # Determine number of points based on step_size
    num_points_segment = max(int(line.length // step_size) + 1, 2)  # At least two points

    distances = np.linspace(0, line.length, num=num_points_segment)
    sampled_points = [line.interpolate(distance) for distance in distances]

    for pt in sampled_points:
        x, y = pt.x, pt.y
        segment_points.append([y, x])  # [y, x] for image coordinates
        segment_labels.append(segment_name)
        segment_name_to_point_indices[segment_name].append(point_index)
        segment_point_coords[point_index] = {'x': x, 'y': y, 'segment_name': segment_name}
        segment_point_indices.append(point_index)
        point_index += 1

print(f"Total sampled points: {len(segment_point_indices)}")

if len(segment_point_indices) == 0:
    raise ValueError("No segment points were sampled. Please check the segments and branch points.")

# Create the KDTree
segment_tree = cKDTree(segment_points)
print("KDTree constructed.")

# --- Step 9: Assign Positive Pixels to the Closest Segment Points ---
print("Assigning positive pixels to closest segment points...")
positive_pixels_all = np.argwhere(binary_image)

print(f"Total positive pixels before filtering: {len(positive_pixels_all)}")

# Query the KDTree
distances, indices = segment_tree.query(positive_pixels_all)

# Find indices where distance <= max_distance
valid_mask = distances <= max_distance
positive_pixels = positive_pixels_all[valid_mask]
indices = indices[valid_mask]
pixel_labels = np.array(segment_labels)[indices]
pixel_point_indices = np.array(segment_point_indices)[indices]

print(f"Total positive pixels within {max_distance} pixels of segments: {len(positive_pixels)}")

if len(positive_pixels) == 0:
    print("No positive pixels within the specified maximum distance. Exiting visualization steps.")
    exit()

# --- Step 10: Mapping Positive Pixels to Segment Points ---
print("Mapping positive pixels to segment points...")
point_positive_pixels = defaultdict(list)
for i, point_idx in enumerate(indices):
    y, x = positive_pixels[i]
    intensity = masked_image[y, x]
    point_positive_pixels[point_idx].append({'y': y, 'x': x, 'intensity': intensity})

print(f"Total unique segment points with assigned pixels: {len(point_positive_pixels)}")

# --- Step 11: Quantify the Signal for Each Segment Point ---
print("Quantifying signal for each segment point...")
point_signal = defaultdict(float)
point_pixel_count = defaultdict(int)

for point_idx, pixels in point_positive_pixels.items():
    total_intensity = sum(pix['intensity'] for pix in pixels)
    count = len(pixels)
    point_signal[point_idx] += total_intensity
    point_pixel_count[point_idx] += count

# Calculate average intensity per point
print("Calculating average intensity per segment point...")
point_average_intensity = {}
for idx in point_signal:
    if point_pixel_count[idx] > 0:
        point_average_intensity[idx] = point_signal[idx] / point_pixel_count[idx]
    else:
        point_average_intensity[idx] = 0

print(f"Calculated average intensities for {len(point_average_intensity)} segment points.")

# --- Step 12: Build the Directed Graph Representing the Hierarchy ---
print("Building directed graph for duct hierarchy...")
G = nx.DiGraph()
for bp_name, bp_data in branch_points.items():
    G.add_node(bp_name, **bp_data)

for segment_name, segment_data in segments.items():
    start_bp = segment_data['start_bp']
    end_bp = segment_data['end_bp']
    G.add_edge(start_bp, end_bp, segment_name=segment_name)

print(f"Directed graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# --- Step 13: Identify Endpoints ---
print("Identifying endpoints...")
endpoints = [node for node in G.nodes if G.out_degree(node) == 0]
print(f"Total Endpoints: {len(endpoints)}")
print(f"Endpoints: {endpoints}")

if len(endpoints) == 0:
    raise ValueError("No endpoints found in the directed graph.")

# --- Step 14: Visualization of Segments with Signals ---
print("Plotting Segments with Assigned Positive Signal...")
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(masked_image, cmap='gray')
ax.set_title('Segments with Assigned Positive Signal')
ax.axis('off')

# Plot all segments in blue
for segment_name in segments.keys():
    indices = [idx for idx, seg_name in enumerate(segment_labels) if seg_name == segment_name]
    coords = [segment_point_coords[idx] for idx in indices]
    x = [coord['x'] for coord in coords]
    y = [coord['y'] for coord in coords]
    ax.plot(x, y, 'b-', linewidth=2)

# Highlight segments with positive signals in red
segments_with_signal = set(pixel_labels)
for segment_name in segments_with_signal:
    indices = [idx for idx, seg_name in enumerate(segment_labels) if seg_name == segment_name]
    coords = [segment_point_coords[idx] for idx in indices]
    x = [coord['x'] for coord in coords]
    y = [coord['y'] for coord in coords]
    ax.plot(x, y, 'r-', linewidth=3)

plt.tight_layout()
plt.show()

# --- Step 15: Visualization of Positive Pixels Assigned to Segments ---
print("Plotting Positive Pixels Assigned to Segments...")
# Create a color map for segments
unique_segments = list(set(segment_labels))
cmap = cm.get_cmap('jet', len(unique_segments))
segment_color_dict = {segment_name: cmap(i) for i, segment_name in enumerate(unique_segments)}

plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(masked_image, cmap='gray')
ax.set_title('Positive Pixels Assigned to Segments')
ax.axis('off')

# Plot the positive pixels, colored by assigned segment
for segment_name in segments_with_signal:
    mask_pixels = pixel_labels == segment_name
    pixels = positive_pixels[mask_pixels]
    if len(pixels) > 0:
        ax.scatter(pixels[:, 1], pixels[:, 0], s=1, color=segment_color_dict[segment_name], label=segment_name)

# To prevent cluttering the legend, limit the number of labels
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), markerscale=10, fontsize='small', bbox_to_anchor=(1.05, 1),
          loc='upper left')

plt.tight_layout()
plt.show()

# --- Step 16: Visualization of Segments Colored by Average Intensity ---
print("Plotting Segments Colored by Average Intensity...")
# Map the average intensities to colors
intensities = list(point_average_intensity.values())
if intensities:
    norm = colors.Normalize(vmin=min(intensities), vmax=max(intensities))
else:
    norm = colors.Normalize(vmin=0, vmax=1)
cmap_hot = cm.get_cmap('hot')
sm = cm.ScalarMappable(cmap=cmap_hot, norm=norm)
sm.set_array([])

plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(masked_image, cmap='gray')
ax.set_title('Segments Colored by Average Intensity')
ax.axis('off')

# Create a list of lines and colors
lines = []
colors_list = []

# Group points by segments
segment_point_groups = defaultdict(list)
for idx, seg_name in zip(segment_point_indices, segment_labels):
    segment_point_groups[seg_name].append(idx)

for segment_name, point_indices in segment_point_groups.items():
    coords = [segment_point_coords[idx] for idx in point_indices]
    x = [coord['x'] for coord in coords]
    y = [coord['y'] for coord in coords]
    avg_intensities = [point_average_intensity.get(idx, 0) for idx in point_indices]
    colors_segment = [sm.to_rgba(intensity) for intensity in avg_intensities]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments_plot = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments_plot, colors=colors_segment, linewidths=3)
    ax.add_collection(lc)

# Add the colorbar
plt.colorbar(sm, ax=ax, label='Average Intensity')

plt.tight_layout()
plt.show()

# --- Step 17: Processing and Visualizing Intensity Profiles Along Paths to Endpoints ---
print("Processing and Visualizing Intensity Profiles Along Paths to Endpoints...")

def process_endpoint(endpoint, G, segment_to_indices, segment_point_coords,
                    point_average_intensity, segments, step_size):
    """
    Processes a single endpoint to extract intensity profiles.
    Returns a dictionary with intensity and binary profiles.
    """
    path_nodes = []
    current_node = endpoint
    visited_nodes = set()

    # Traverse from endpoint to root
    while True:
        if current_node in visited_nodes:
            print(f"Warning: Loop detected starting at node {current_node}")
            break
        visited_nodes.add(current_node)
        path_nodes.append(current_node)
        predecessors = list(G.predecessors(current_node))
        if not predecessors:
            break  # Reached the root
        current_node = predecessors[0]  # Assuming one parent

    path_nodes = path_nodes[::-1]  # From root to endpoint

    full_x = []
    full_y = []
    intensities_along_path = []

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            print(f"Error: No edge data for {u} -> {v}")
            continue
        segment_name = edge_data['segment_name']

        # Get precomputed point indices
        point_indices = segment_to_indices.get(segment_name, [])
        if not point_indices:
            print(f"Warning: No points found for segment {segment_name}")
            continue

        # Extract coordinates and intensities
        coords = [segment_point_coords[idx] for idx in point_indices]
        x_coords = [coord['x'] for coord in coords]
        y_coords = [coord['y'] for coord in coords]
        avg_intensities = [point_average_intensity.get(idx, 0) for idx in point_indices]

        segment_data = segments[segment_name]

        # Check the order of the segment
        if segment_data['start_bp'] == u and segment_data['end_bp'] == v:
            pass  # Correct order
        elif segment_data['start_bp'] == v and segment_data['end_bp'] == u:
            x_coords = x_coords[::-1]
            y_coords = y_coords[::-1]
            avg_intensities = avg_intensities[::-1]
        else:
            print(f"Error: Segment {segment_name} does not connect {u} to {v}.")
            continue

        # Exclude the last point to avoid duplicates, except for the final segment
        if i < len(path_nodes) - 2:
            full_x.extend(x_coords[:-1])
            full_y.extend(y_coords[:-1])
            intensities_along_path.extend(avg_intensities[:-1])
        else:
            full_x.extend(x_coords)
            full_y.extend(y_coords)
            intensities_along_path.extend(avg_intensities)

    if not full_x:
        print(f"No valid path found for endpoint {endpoint}.")
        return None

    # Convert to NumPy arrays for efficient computation
    path_points = np.column_stack((full_x, full_y))
    path_line = LineString(path_points)
    num_samples = int(np.ceil(path_line.length / step_size)) + 1  # Number of samples based on step_size
    sampled_distances = np.linspace(0, path_line.length, num=num_samples)
    sampled_points = [path_line.interpolate(distance) for distance in sampled_distances]

    # Compute intensities at sampled points
    sampled_intensities = []
    for pt in sampled_points:
        x, y = pt.x, pt.y
        # Option 1: Nearest point intensity
        distance, idx = segment_tree.query([y, x], k=1)
        sampled_intensities.append(point_average_intensity.get(segment_point_indices[idx], 0))
        # Option 2: Interpolate intensity if necessary
        # (Implement interpolation if needed)

    # Create binary readout based on the mean intensity
    intensity_threshold = np.mean(sampled_intensities)
    path_binary_values = (np.array(sampled_intensities) > intensity_threshold).astype(int)

    return {
        'intensities': np.array(sampled_intensities),
        'binary': path_binary_values,
        'length': path_line.length,
        'endpoint': endpoint,
        'distances': sampled_distances,
        'path_nodes': path_nodes
    }

# --- Step 17a: Assign Tracks to Branches ---
print("Assigning tracks to their respective branches...")

def assign_branch(path_nodes, G):
    """
    Assigns a branch based on the path nodes.
    Here, the branch is determined by the first segment.
    """
    if len(path_nodes) < 2:
        return None
    u, v = path_nodes[0], path_nodes[1]
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        return None
    return edge_data.get('segment_name', None)

# Assign branches to each path
path_branches = []
for path_nodes in [res['path_nodes'] for res in []]:  # Placeholder; will update later
    branch = assign_branch(path_nodes, G)
    path_branches.append(branch)

# Note: The actual assignment will be done after processing endpoints

# --- Step 18: Select and Assign Endpoints to Branches ---
print("Selecting endpoints based on single_endpoint_tracks indices...")
selected_endpoints = endpoints

print(f"Total selected endpoints: {len(selected_endpoints)}")

if len(selected_endpoints) == 0:
    raise ValueError("No valid endpoints selected. Please check single_endpoint_tracks indices.")

# --- Step 17b: Assign Branches after Processing Endpoints ---
print("Processing endpoints in parallel and assigning branches...")

# Process endpoints in parallel
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_endpoint)(
        endpoint, G, segment_name_to_point_indices, segment_point_coords,
        point_average_intensity, segments, step_size
    ) for endpoint in selected_endpoints
)

# Filter out None results
results = [res for res in results if res is not None]
print(f"Processed {len(results)} endpoints successfully.")


# --- Step 23: Heatmap of All Intensity Profiles ---
print("Plotting Heatmap for All Intensity Profiles...")

# Determine the maximum path length
max_length = max(len(intensity) for intensity in [res['intensities'] for res in results])
print(f"Maximum path length: {max_length}")

# Initialize a 2D array with NaNs
intensity_matrix = np.full((len(results), max_length), np.nan)

# Populate the intensity_matrix with the intensity profiles
for i, intensity in enumerate([res['intensities'] for res in results]):
    intensity_matrix[i, :len(intensity)] = intensity

# Plot the heatmap
plt.figure()
sns.heatmap(
    intensity_matrix,
    cmap='hot',
    cbar=True,
    xticklabels=10,
    yticklabels=False,
    mask=np.isnan(intensity_matrix),
    linewidths=0,
    linecolor='gray'
)
plt.title('Heatmap of Intensity Profiles Across All Paths')
plt.xlabel('Path Position')
plt.ylabel('Path Index')
plt.tight_layout()
plt.show()

print("Heatmap for all intensity profiles plotted successfully.")
