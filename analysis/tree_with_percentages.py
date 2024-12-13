import numpy as np
import matplotlib.pyplot as plt
import json
from skimage import io
from shapely.geometry import LineString, shape
from rasterio.features import rasterize
import networkx as nx
from matplotlib import cm, colors

ecad_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0003.tif'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max.lif - TileScan 1 Merged_Processed001_duct.geojson'
annotated_tracks_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\890_annotations.json'

# Parameters
threshold_value = 500

# Step 1: Load the raw Ecad channel image
ecad_image = io.imread(ecad_image_path)

# Step 2: Load and rasterize the duct border annotations to create a mask
with open(duct_borders_path, 'r') as f:
    duct_borders = json.load(f)

shapes = []
for feature in duct_borders['features']:
    geometry = feature['geometry']
    shapely_geom = shape(geometry)
    shapes.append((shapely_geom, 1))  # Assign a value of 1 inside the polygon

mask = rasterize(
    shapes,
    out_shape=ecad_image.shape,
    fill=0,
    dtype=np.uint8,
    all_touched=False
)

masked_image = ecad_image.copy()
masked_image[mask == 0] = 0

# Step 3: Threshold the image to detect positive signal
binary_image = masked_image > threshold_value

# Step 4: Load the annotated tracks (duct segments and branch points)
with open(annotated_tracks_path, 'r') as f:
    annotated_tracks = json.load(f)

system = annotated_tracks['duct_systems'][2]
branch_points = system['branch_points']
segments = system['segments']

# Step 5: Rasterize all segments at once
segment_id_mapping = {}
segment_shapes = []
for idx, (segment_name, segment_data) in enumerate(segments.items(), start=1):
    segment_id = idx
    segment_id_mapping[segment_id] = segment_name

    start_bp = branch_points[segment_data['start_bp']]
    end_bp = branch_points[segment_data['end_bp']]
    points = [(start_bp['x'], start_bp['y'])]

    for point in segment_data.get('internal_points', []):
        points.append((point['x'], point['y']))

    points.append((end_bp['x'], end_bp['y']))

    line = LineString(points)
    buffered_line = line.buffer(5)

    segment_shapes.append((buffered_line, segment_id))

# Rasterize all segments together
segment_mask = rasterize(
    shapes=segment_shapes,
    out_shape=ecad_image.shape,
    fill=0,
    dtype=np.uint16,
    all_touched=True
)

# Apply the duct mask to the segment mask
segment_mask[mask == 0] = 0

# Flatten the arrays for vectorized operations
flat_segment_mask = segment_mask.flatten()
flat_binary_image = binary_image.flatten()

# Keep only the pixels that belong to segments
valid_indices = flat_segment_mask > 0
flat_segment_mask = flat_segment_mask[valid_indices]
flat_binary_image = flat_binary_image[valid_indices]

# Calculate total and positive pixels per segment using bincount
total_pixels_per_segment = np.bincount(flat_segment_mask)
positive_pixels_per_segment = np.bincount(
    flat_segment_mask, weights=flat_binary_image.astype(np.uint8))

# Step 6: Calculate the percentage of positive pixels for each segment
segment_percentages = {}
for segment_id in range(1, max(segment_id_mapping.keys()) + 1):
    total_pixels = total_pixels_per_segment[segment_id] if segment_id < len(
        total_pixels_per_segment) else 0
    positive_pixels = positive_pixels_per_segment[segment_id] if segment_id < len(
        positive_pixels_per_segment) else 0

    if total_pixels > 0:
        percentage = (positive_pixels / total_pixels) * 100
    else:
        percentage = 0

    segment_name = segment_id_mapping[segment_id]
    segment_percentages[segment_name] = percentage

# Step 7: Build the tree structure from segments and branch points
G = nx.Graph()
for segment_name, segment_data in segments.items():
    start_bp = segment_data['start_bp']
    end_bp = segment_data['end_bp']
    G.add_edge(start_bp, end_bp, segment_name=segment_name)

# Step 8: Plot the tree with segments colored according to positive percentages
if not nx.is_tree(G):
    T = nx.minimum_spanning_tree(G)
else:
    T = G


def hierarchy_pos(G, root=None, vert_gap=0.2, vert_loc=0, x_start=0):
    """
    Positions nodes in a hierarchical layout, allowing overlapping x-coordinates at different levels.

    Parameters:
    - G: The graph (must be a tree).
    - root: The root node of the tree.
    - vert_gap: Vertical gap between levels of the tree.
    - vert_loc: Vertical location of the root.
    - x_start: Starting horizontal position.

    Returns:
    - pos: A dictionary mapping nodes to positions (x, y).
    """
    if not nx.is_tree(G):
        raise TypeError('This function requirs a tree graph.')
    if root is None:
        root = list(G.nodes)[0]
    pos = {}
    next_x = [x_start-1]

    def recurse(node, depth, parent=None):
        children = list(G.neighbors(node))
        if parent is not None and parent in children:
            children.remove(parent)
        if len(children) == 0:
            # Leaf node
            pos[node] = (next_x[0], vert_loc - depth * vert_gap)
            next_x[0] += 1  # Move to the next horizontal position
        else:
            # Internal node
            child_x = []
            for child in children:
                recurse(child, depth + 1, node)
                child_x.append(pos[child][0])
            # Position this node at the center of its children
            pos[node] = ((min(child_x) + max(child_x)) / 2, vert_loc - depth * vert_gap)

    recurse(root, 0)
    return pos

root_node = list(T.nodes)[0]
pos = hierarchy_pos(T, root=root_node)

edge_colors = []
for u, v in T.edges():
    segment_name = T.edges[u, v]['segment_name']
    percentage = segment_percentages.get(segment_name, 0)
    edge_colors.append(percentage)

max_percentage = max(edge_colors)
colors_with_black = np.vstack([[0, 0, 0, 1], cm.get_cmap('coolwarm')(np.linspace(0, 1, 256))])
cmap = colors.ListedColormap(colors_with_black)

# Normalize and apply the new colormap
norm = colors.Normalize(vmin=0, vmax=max_percentage)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

edge_colors_mapped = [cmap(norm(percentage)) for percentage in edge_colors]

plt.figure(figsize=(12, 8))
ax = plt.gca()

nx.draw(T, pos=pos, with_labels=False, node_size=0, node_color='none', edge_color=edge_colors_mapped, width=2, ax=ax)

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('% Positive Pixels')

plt.title('Tree Colored by % Positive Pixels per Segment')
plt.axis('off')
plt.show()
