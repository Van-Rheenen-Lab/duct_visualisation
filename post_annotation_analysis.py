import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import io, filters, morphology
from shapely.geometry import shape, LineString
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from collections import defaultdict
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import networkx as nx
import matplotlib.gridspec as gridspec


ecad_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
annotated_tracks_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'

# Set a maximum distance threshold
max_distance = 25
override_threshold = 800 # make None for automatic thresholding

single_endpoint_tracks = [50, 100, 850, 901, 950, 980, 1001, 1050]

# Step 1: Load the raw Ecad channel image (tif image)
ecad_image = io.imread(ecad_image_path)

# Step 2: Load the duct border annotations
with open(duct_borders_path, 'r') as f:
    duct_borders = json.load(f)

# Create an empty mask with the same shape as the image
# Prepare a list of shapes for rasterization
shapes = []
for feature in duct_borders['features']:
    geometry = feature['geometry']
    shapely_geom = shape(geometry)
    shapes.append((shapely_geom, 1))  # Assign a value of 1 inside the polygon

# Rasterize the shapes onto the mask
mask = rasterize(
    shapes,
    out_shape=ecad_image.shape,
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
if not override_threshold:
    threshold_value = filters.threshold_otsu(ecad_image)  # Consider only masked pixels
    print(f"Threshold value: {threshold_value}")
else:
    threshold_value = override_threshold

binary_image = masked_image > threshold_value

# Step 6: Load the annotated tracks (duct segments and branch points)
with open(annotated_tracks_path, 'r') as f:
    annotated_tracks = json.load(f)

# Assuming you are working with the first duct system
system = annotated_tracks['duct_systems'][0]
branch_points = system['branch_points']
segments = system['segments']

# Align the coordinate system if necessary
# Find minimum x and y values
min_x = min([bp['x'] for bp in branch_points.values()])
min_y = min([bp['y'] for bp in branch_points.values()])

for segment_data in segments.values():
    for point in segment_data.get('internal_points', []):
        min_x = min(min_x, point['x'])
        min_y = min(min_y, point['y'])

# Calculate offsets
x_offset = -min_x if min_x < 0 else 0
y_offset = -min_y if min_y < 0 else 0

# Apply offsets to all coordinates
for bp in branch_points.values():
    bp['x'] += x_offset
    bp['y'] += y_offset

for segment_data in segments.values():
    for point in segment_data.get('internal_points', []):
        point['x'] += x_offset
        point['y'] += y_offset

# Step 7: Build a spatial index (KDTree) for the segments
segment_points = []
segment_labels = []
segment_point_indices = []  # To keep track of point indices
segment_point_coords = {}   # To store coordinates for each point along segments

point_index = 0
for segment_name, segment_data in segments.items():
    # Get the start and end branch point names
    start_bp_name = segment_data['start_bp']
    end_bp_name = segment_data['end_bp']

    # Get the coordinates of the start and end branch points
    start_bp = branch_points[start_bp_name]
    end_bp = branch_points[end_bp_name]

    # Build the list of points for the segment
    points = []

    # Start with start_bp
    points.append((start_bp['x'], start_bp['y']))

    # Add internal points if any
    for point in segment_data.get('internal_points', []):
        points.append((point['x'], point['y']))

    # End with end_bp
    points.append((end_bp['x'], end_bp['y']))

    # Store the coordinates for plotting
    x_coords = [pt[0] for pt in points]
    y_coords = [pt[1] for pt in points]

    # Create a LineString for the segment
    line = LineString(points)

    # Sample points along the LineString
    num_points = 100  # Increase the number of points for better mapping
    distances = np.linspace(0, line.length, num_points)
    sampled_points = [line.interpolate(distance) for distance in distances]

    for pt in sampled_points:
        x, y = pt.x, pt.y
        # Append to segment_points and segment_labels
        segment_points.append([y, x])  # Note: [y, x] for image coordinates
        segment_labels.append(segment_name)
        segment_point_indices.append(point_index)
        segment_point_coords[point_index] = {'x': x, 'y': y, 'segment_name': segment_name}
        point_index += 1

# Create the KDTree
segment_tree = cKDTree(segment_points)

# Step 8: For each positive pixel, assign it to the closest point along the segments
positive_pixels_all = np.argwhere(binary_image)

# Query the KDTree
distances, indices = segment_tree.query(positive_pixels_all)

valid_indices = distances <= max_distance

# Filter positive_pixels and pixel_labels based on the distance threshold
positive_pixels = positive_pixels_all[valid_indices]
indices = indices[valid_indices]
pixel_labels = np.array(segment_labels)[indices]
pixel_point_indices = np.array(segment_point_indices)[indices]

print(f"Total positive pixels: {len(positive_pixels_all)}")
print(f"Positive pixels within {max_distance} pixels of segments: {len(positive_pixels)}")

# Build a mapping from segment point indices to positive pixels assigned to them
point_positive_pixels = defaultdict(list)
for i, point_idx in enumerate(pixel_point_indices):
    y, x = positive_pixels[i]
    intensity = masked_image[y, x]
    point_positive_pixels[point_idx].append({'y': y, 'x': x, 'intensity': intensity})

# Now, let's first plot the ducts with all the binarized signal in one image.
plt.figure(figsize=(10, 10))
plt.imshow(masked_image, cmap='gray')
plt.imshow(binary_image, cmap='Reds', alpha=0.5)
plt.title('Masked Ecad Image with Binarized Signal Overlay')
plt.axis('off')


# Now, color the parts of the segments that the signal is assigned to red
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(masked_image, cmap='gray')
ax.set_title('Segments with Assigned Positive Signal')
ax.axis('off')

# Plot all segments in blue with intermediate points
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


# Optional: Plot the positive pixels assigned to each segment
# Create a color map for segments
unique_segments = list(set(segment_labels))
cmap = cm.get_cmap('jet', len(unique_segments))
segment_color_dict = {segment_name: cmap(i) for i, segment_name in enumerate(unique_segments)}

# Create a figure
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(masked_image, cmap='gray')
ax.set_title('Positive Pixels Assigned to Segments')
ax.axis('off')

# Plot the positive pixels, colored by assigned segment
for segment_name in segments_with_signal:
    mask_pixels = pixel_labels == segment_name
    pixels = positive_pixels[mask_pixels]
    ax.scatter(pixels[:, 1], pixels[:, 0], s=1, color=segment_color_dict[segment_name], label=segment_name)

plt.legend(markerscale=10, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')


# Now, analyze the positive signals along the ducts.

# Step 9: Quantify the signal for each segment point
point_signal = defaultdict(float)
point_pixel_count = defaultdict(int)

for point_idx in point_positive_pixels:
    pixels = point_positive_pixels[point_idx]
    for pix in pixels:
        intensity = pix['intensity']
        point_signal[point_idx] += intensity
        point_pixel_count[point_idx] += 1

# Calculate average intensity per point
point_average_intensity = {}
for point_idx in point_signal:
    if point_pixel_count[point_idx] > 0:
        point_average_intensity[point_idx] = point_signal[point_idx] / point_pixel_count[point_idx]
    else:
        point_average_intensity[point_idx] = 0

# Step 10: Report the quantified signal along the ducts
# (This can be modified as needed)

# Visualization of average intensity along segments
# Map the average intensities to colors
intensities = list(point_average_intensity.values())
if intensities:
    norm = colors.Normalize(vmin=min(intensities), vmax=max(intensities))
else:
    norm = colors.Normalize(vmin=0, vmax=1)
cmap = cm.get_cmap('hot')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
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


# Visualization from endpoints to the root with a binary readout

# Step 1: Build the directed graph representing the hierarchy
G = nx.DiGraph()

# Add nodes
for bp_name, bp_data in branch_points.items():
    G.add_node(bp_name, **bp_data)

# Add edges from parent to child (start_bp to end_bp)
for segment_name, segment_data in segments.items():
    start_bp = segment_data['start_bp']
    end_bp = segment_data['end_bp']
    G.add_edge(start_bp, end_bp, segment_name=segment_name)

# Step 2: Identify endpoints (nodes with zero out-degree)
endpoints = [node for node in G.nodes if G.out_degree(node) == 0]
print("Endpoints:", endpoints)

# Process a couple of endpoints first
selected_endpoints = [endpoints[i] for i in single_endpoint_tracks ]

for endpoint in selected_endpoints:
    print(f"\nProcessing endpoint: {endpoint}")

    # Step 3: Traverse from the endpoint back to the root
    path_nodes = []
    current_node = endpoint
    while True:
        path_nodes.append(current_node)
        predecessors = list(G.predecessors(current_node))  # Edges from parent to child
        if not predecessors:
            break  # Reached the root
        current_node = predecessors[0]  # Assuming one parent

    # Reverse the path to have it from root to endpoint
    path_nodes = path_nodes[::-1]
    print(f"Path from root to {endpoint}: {path_nodes}")

    # Step 4: Collect coordinates and intensities along the path
    full_x = []
    full_y = []
    intensities_along_path = []

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)  # Edge from parent (u) to child (v)
        segment_name = edge_data['segment_name']

        # Get point indices for this segment
        point_indices = [idx for idx, seg_name in enumerate(segment_labels) if seg_name == segment_name]
        coords = [segment_point_coords[idx] for idx in point_indices]

        x_coords = [coord['x'] for coord in coords]
        y_coords = [coord['y'] for coord in coords]
        avg_intensities = [point_average_intensity.get(idx, 0) for idx in point_indices]

        segment_data = segments[segment_name]

        # Check the order of the segment
        if segment_data['start_bp'] == u and segment_data['end_bp'] == v:
            pass  # Order is correct
        elif segment_data['start_bp'] == v and segment_data['end_bp'] == u:
            x_coords = x_coords[::-1]
            y_coords = y_coords[::-1]
            avg_intensities = avg_intensities[::-1]
        else:
            print("Error: Segment does not connect the expected branch points.")

        # Exclude the last point to avoid duplicates
        if i < len(path_nodes) - 2:
            full_x.extend(x_coords[:-1])
            full_y.extend(y_coords[:-1])
            intensities_along_path.extend(avg_intensities[:-1])
        else:
            full_x.extend(x_coords)
            full_y.extend(y_coords)
            intensities_along_path.extend(avg_intensities)

    # Check if the path is valid
    if not full_x:
        print(f"No path found from root to {endpoint}.")
        continue

    # Step 5: Sample points along the path and extract average intensities
    # Create a LineString from the collected coordinates
    path_points = list(zip(full_x, full_y))
    path_line = LineString(path_points)

    # Sample points along the LineString
    num_samples = 100  # Adjust as needed
    distances = np.linspace(0, path_line.length, num_samples)
    sampled_points = [path_line.interpolate(distance) for distance in distances]

    # Interpolate intensities along the path
    from scipy.interpolate import interp1d

    # Get cumulative distances along the path
    cumulative_distances = [0]
    for i in range(1, len(path_points)):
        prev_point = path_points[i - 1]
        curr_point = path_points[i]
        dist = np.hypot(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        cumulative_distances.append(cumulative_distances[-1] + dist)

    # Interpolate the intensities
    intensity_interpolator = interp1d(cumulative_distances, intensities_along_path, kind='linear', fill_value="extrapolate")
    sampled_intensities = intensity_interpolator(distances)

    # Create a binary readout based on a threshold (e.g., mean intensity)
    intensity_threshold = np.mean(sampled_intensities)
    path_binary_values = (sampled_intensities > intensity_threshold).astype(int)

    # Step 6: Visualize the intensity trace and the binary readout

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2)

    # Plot the path on the image
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(masked_image, cmap='gray')
    ax0.set_title(f'Path from Root to Endpoint {endpoint}')
    ax0.axis('off')

    # Plot the path
    path_x = [pt.x for pt in sampled_points]
    path_y = [pt.y for pt in sampled_points]
    ax0.plot(path_x, path_y, 'r-', linewidth=2)

    # Plot the pixel values along the path
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(sampled_intensities, 'b-')
    ax1.set_title('Average Intensity Along the Path')
    ax1.set_xlabel('Sample Point')
    ax1.set_ylabel('Average Intensity')

    # Plot the binary readout along the path
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(path_binary_values, 'r-')
    ax2.set_title('Binary Readout Along the Path')
    ax2.set_xlabel('Sample Point')
    ax2.set_ylabel('Binary Value')

    plt.tight_layout()


import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from sklearn.cluster import KMeans
from collections import Counter
from joblib import Parallel, delayed
import networkx as nx  # Assuming G is a NetworkX graph


# Define a function to process a single endpoint
def process_endpoint(endpoint, G, segment_labels, segment_point_coords,
                     point_average_intensity, segments):
    # Step 3: Traverse from the endpoint back to the root
    path_nodes = []
    current_node = endpoint
    visited_nodes = set()

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

    path_nodes = path_nodes[::-1]

    # Step 4: Collect coordinates and intensities along the path
    full_x = []
    full_y = []
    intensities_along_path = []

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        segment_name = edge_data['segment_name']

        # Get point indices for this segment
        point_indices = [idx for idx, seg_name in enumerate(segment_labels) if seg_name == segment_name]
        coords = [segment_point_coords[idx] for idx in point_indices]

        x_coords = [coord['x'] for coord in coords]
        y_coords = [coord['y'] for coord in coords]
        avg_intensities = [point_average_intensity.get(idx, 0) for idx in point_indices]

        segment_data = segments[segment_name]

        # Check the order of the segment
        if segment_data['start_bp'] == u and segment_data['end_bp'] == v:
            pass  # Order is correct
        elif segment_data['start_bp'] == v and segment_data['end_bp'] == u:
            x_coords.reverse()
            y_coords.reverse()
            avg_intensities.reverse()
        else:
            print("Error: Segment does not connect the expected branch points.")

        # Exclude the last point to avoid duplicates
        if i < len(path_nodes) - 2:
            full_x.extend(x_coords[:-1])
            full_y.extend(y_coords[:-1])
            intensities_along_path.extend(avg_intensities[:-1])
        else:
            full_x.extend(x_coords)
            full_y.extend(y_coords)
            intensities_along_path.extend(avg_intensities)

    if not full_x:
        print(f"No path found from root to {endpoint}.")
        return None  # Early exit if path is invalid

    # Step 5: Sample points along the path and extract average intensities
    path_points = list(zip(full_x, full_y))
    path_line = LineString(path_points)

    # Sample points along the LineString
    num_samples = max(int(path_line.length), 2)  # Ensure at least two samples
    distances = np.linspace(0, path_line.length, num_samples)
    sampled_points = [path_line.interpolate(distance) for distance in distances]

    # Compute cumulative distances
    diffs = np.diff(path_points, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Interpolate intensities using NumPy
    sampled_intensities = np.interp(distances, cumulative_distances, intensities_along_path)

    # Create binary readout based on the mean intensity
    intensity_threshold = sampled_intensities.mean()
    path_binary_values = (sampled_intensities > intensity_threshold).astype(int)

    return {
        'intensities': sampled_intensities,
        'binary': path_binary_values,
        'length': path_line.length,
        'endpoint': endpoint,
        'distances': distances,
        'path_nodes': path_nodes
    }

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_endpoint)(
        endpoint, G, segment_labels, segment_point_coords,
        point_average_intensity, segments
    ) for endpoint in endpoints
)

# Filter out None results
results = [res for res in results if res is not None]

# Unpack results
all_paths_intensities = [res['intensities'] for res in results]
all_paths_binary = [res['binary'] for res in results]
all_paths_lengths = [res['length'] for res in results]
all_endpoints = [res['endpoint'] for res in results]
all_distances = [res['distances'] for res in results]
all_path_nodes = [res['path_nodes'] for res in results]

# Handle Non-Normalized Data
max_length = max(len(intensities) for intensities in all_paths_intensities)

# Plot non-normalized intensities
plt.figure(figsize=(12, 8))
for intensities, distances, endpoint in zip(all_paths_intensities, all_distances, all_endpoints):
    plt.plot(distances, intensities, label=f'Path to {endpoint}')
plt.title('Intensity Profiles Along Paths to Endpoints (Non-Normalized)')
plt.xlabel('Distance Along Path')
plt.ylabel('Intensity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot non-normalized binary readouts
plt.figure(figsize=(12, 8))
for idx, (binary_values, distances) in enumerate(zip(all_paths_binary, all_distances)):
    plt.plot(distances, binary_values + idx * 2, label=f'Path to {all_endpoints[idx]}')  # Offset for clarity
plt.title('Binary Readouts Along Paths to Endpoints (Non-Normalized)')
plt.xlabel('Distance Along Path')
plt.ylabel('Binary Readout')
plt.yticks([])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Now, for normalized data
# Identify common starting nodes
start_nodes = [path_nodes[0] for path_nodes in all_path_nodes]
common_root = Counter(start_nodes).most_common(1)[0][0]

# Trim the common segments from paths
trimmed_intensities = []
trimmed_distances = []
for intensities, distances, path_nodes in zip(all_paths_intensities, all_distances, all_path_nodes):
    # Find the index where the path deviates from the common root
    deviation_index = next((i for i, node in enumerate(path_nodes) if node != common_root), len(path_nodes))

    if deviation_index >= len(path_nodes):
        trimmed_intensities.append(intensities)
        trimmed_distances.append(distances)
        continue

    # Compute the corresponding sample index
    sample_index = int(deviation_index / len(path_nodes) * len(intensities))
    trimmed_intensities.append(intensities[sample_index:])
    trimmed_distances.append(distances[sample_index:] - distances[sample_index])

# Normalize trimmed paths to the same length
max_trimmed_length = max(len(intensities) for intensities in trimmed_intensities)
aligned_trimmed_intensities = []
for intensities in trimmed_intensities:
    if len(intensities) == max_trimmed_length:
        aligned_trimmed_intensities.append(intensities)
    else:
        aligned = np.interp(
            np.linspace(0, 1, max_trimmed_length),
            np.linspace(0, 1, len(intensities)),
            intensities
        )
        aligned_trimmed_intensities.append(aligned)

aligned_trimmed_intensities = np.array(aligned_trimmed_intensities)

# Compute mean and standard deviation
mean_trimmed_intensity = aligned_trimmed_intensities.mean(axis=0)
std_trimmed_intensity = aligned_trimmed_intensities.std(axis=0)

# Plot the mean intensity with error bars for trimmed paths
plt.figure(figsize=(10, 6))
plt.plot(mean_trimmed_intensity, label='Mean Intensity (Trimmed)')
plt.fill_between(
    range(max_trimmed_length),
    mean_trimmed_intensity - std_trimmed_intensity,
    mean_trimmed_intensity + std_trimmed_intensity,
    alpha=0.3,
    label='Std Dev'
)
plt.title('Mean Intensity Along Trimmed Paths to Endpoints (Normalized)')
plt.xlabel('Normalized Path Length')
plt.ylabel('Intensity')
plt.legend()

# Align untrimmed paths
aligned_intensities = []
for intensities in all_paths_intensities:
    if len(intensities) == max_length:
        aligned_intensities.append(intensities)
    else:
        aligned = np.interp(
            np.linspace(0, 1, max_length),
            np.linspace(0, 1, len(intensities)),
            intensities
        )
        aligned_intensities.append(aligned)

aligned_intensities = np.array(aligned_intensities)

# Compute mean and standard deviation
mean_intensity = aligned_intensities.mean(axis=0)
std_intensity = aligned_intensities.std(axis=0)

# Plot the mean intensity with error bars
plt.figure(figsize=(10, 6))
plt.plot(mean_intensity, label='Mean Intensity (Untrimmed)')
plt.fill_between(
    range(max_length),
    mean_intensity - std_intensity,
    mean_intensity + std_intensity,
    alpha=0.3,
    label='Std Dev'
)
plt.title('Mean Intensity Along Paths to Endpoints (Normalized)')
plt.xlabel('Normalized Path Length')
plt.ylabel('Intensity')
plt.legend()

# Compare trimmed and untrimmed mean intensities
plt.figure(figsize=(10, 6))
plt.plot(
    np.linspace(0, 1, max_length),
    mean_intensity,
    label='Untrimmed'
)
plt.plot(
    np.linspace(0, 1, max_trimmed_length),
    mean_trimmed_intensity,
    label='Trimmed'
)
plt.title('Comparison of Mean Intensities (Normalized)')
plt.xlabel('Normalized Path Length')
plt.ylabel('Intensity')
plt.legend()

# Clustering the intensity profiles of trimmed paths
num_clusters = 4  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(aligned_trimmed_intensities)
cluster_labels = kmeans.labels_

# Plot the clustered intensity profiles
plt.figure(figsize=(12, 8))
for intensities, label in zip(aligned_trimmed_intensities, cluster_labels):
    plt.plot(intensities, alpha=0.5, label=f'Cluster {label}')
plt.title('Clustered Intensity Profiles (Trimmed Paths)')
plt.xlabel('Normalized Path Length')
plt.ylabel('Intensity')
plt.legend()

# Plot the mean profile for each cluster
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    if len(cluster_indices) == 0:
        continue  # Skip empty clusters
    cluster_intensities = aligned_trimmed_intensities[cluster_indices]
    mean_cluster_intensity = cluster_intensities.mean(axis=0)
    plt.plot(mean_cluster_intensity, label=f'Cluster {cluster}')
plt.title('Mean Intensity Profiles for Each Cluster (Trimmed Paths)')
plt.xlabel('Normalized Path Length')
plt.ylabel('Intensity')
plt.legend()

plt.tight_layout()
plt.show()