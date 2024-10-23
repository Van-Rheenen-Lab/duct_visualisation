import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import io, filters, morphology
from shapely.geometry import shape
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from collections import defaultdict
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import LineCollection

# Define file paths (update these paths according to your directory structure)
ecad_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
annotated_tracks_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'

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
threshold_value = filters.threshold_otsu(masked_image[mask])  # Consider only masked pixels
binary_image = masked_image > threshold_value

# Remove the binary cleaning step as per your request
# binary_image = morphology.binary_opening(binary_image)
# binary_image = morphology.remove_small_objects(binary_image, min_size=50)

# Step 6: Load the annotated tracks (duct segments and branch points)
with open(annotated_tracks_path, 'r') as f:
    annotated_tracks = json.load(f)

segments = annotated_tracks['duct_systems'][0]['segments']
branch_points = annotated_tracks['duct_systems'][0]['branch_points']

# Step 7: Build a spatial index (KDTree) for the segments
segment_points = []
segment_labels = []
segment_coordinates = {}  # To store coordinates for each segment

for segment_name, segment_data in segments.items():
    # Get the start and end branch point names
    start_bp_name = segment_data['start_bp']
    end_bp_name = segment_data['end_bp']

    # Get the coordinates of the start and end branch points
    start_bp = branch_points[start_bp_name]
    end_bp = branch_points[end_bp_name]

    # Extract coordinates
    x1 = start_bp['x']
    y1 = start_bp['y']
    x2 = end_bp['x']
    y2 = end_bp['y']

    # Store the coordinates for plotting
    segment_coordinates[segment_name] = {'x': [x1, x2], 'y': [y1, y2]}

    # Optionally, interpolate points along the segment if needed
    num_points = 10  # Number of points to interpolate between start and end
    x_coords = np.linspace(x1, x2, num=num_points)
    y_coords = np.linspace(y1, y2, num=num_points)

    for x, y in zip(x_coords, y_coords):
        segment_points.append([y, x])  # Note: use [y, x] for image coordinates
        segment_labels.append(segment_name)

segment_tree = cKDTree(segment_points)

# Step 8: For each positive pixel, assign it to the closest segment
positive_pixels_all = np.argwhere(binary_image)

# Query the KDTree
distances, indices = segment_tree.query(positive_pixels_all)

# Set a maximum distance threshold
max_distance = 10  # Adjust this threshold as needed
valid_indices = distances <= max_distance

# Filter positive_pixels and pixel_labels based on the distance threshold
positive_pixels = positive_pixels_all[valid_indices]
pixel_labels = np.array(segment_labels)[indices[valid_indices]]

print(f"Total positive pixels: {len(positive_pixels_all)}")
print(f"Positive pixels within {max_distance} pixels of segments: {len(positive_pixels)}")

# Now, let's first plot the ducts with all the binarized signal in one image.
plt.figure(figsize=(10, 10))
plt.imshow(masked_image, cmap='gray')
plt.imshow(binary_image, cmap='Reds', alpha=0.5)
plt.title('Masked Ecad Image with Binarized Signal Overlay')
plt.axis('off')
plt.show()

# Now, color the parts of the segments that the signal is assigned to red
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(masked_image, cmap='gray')
ax.set_title('Segments with Assigned Positive Signal')
ax.axis('off')

# Plot all segments in blue
for segment_name, coords in segment_coordinates.items():
    x = coords['x']
    y = coords['y']
    ax.plot(x, y, 'b-', linewidth=2)

# Highlight segments with positive signals in red
segments_with_signal = set(pixel_labels)
for segment_name in segments_with_signal:
    coords = segment_coordinates[segment_name]
    x = coords['x']
    y = coords['y']
    ax.plot(x, y, 'r-', linewidth=3)

plt.show()

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
plt.show()

# Now, analyze the positive signals along the ducts.

# Step 9: Quantify the signal for each segment
segment_signal = defaultdict(float)
segment_pixel_count = defaultdict(int)

for i, label in enumerate(pixel_labels):
    y, x = positive_pixels[i]
    intensity = masked_image[y, x]
    segment_signal[label] += intensity
    segment_pixel_count[label] += 1

# Calculate average intensity per segment
segment_average_intensity = {}
for label in segment_signal:
    if segment_pixel_count[label] > 0:
        segment_average_intensity[label] = segment_signal[label] / segment_pixel_count[label]
    else:
        segment_average_intensity[label] = 0

# Step 10: Report the quantified signal along the ducts
print("Average Intensity per Segment:")
for segment_name, avg_intensity in segment_average_intensity.items():
    print(f"Segment {segment_name}: Average Intensity = {avg_intensity:.2f}")

# Visualization of average intensity along segments

# Map the average intensities to colors
intensities = list(segment_average_intensity.values())
norm = colors.Normalize(vmin=min(intensities), vmax=max(intensities))
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

for segment_name, coords in segment_coordinates.items():
    x = coords['x']
    y = coords['y']
    avg_intensity = segment_average_intensity.get(segment_name, 0)
    color = sm.to_rgba(avg_intensity)
    lines.append(list(zip(x, y)))
    colors_list.append(color)

# Create a LineCollection
lc = LineCollection(lines, colors=colors_list, linewidths=3)

# Add the collection to the plot
ax.add_collection(lc)

# Add the colorbar
plt.colorbar(sm, ax=ax, label='Average Intensity')
plt.show()
