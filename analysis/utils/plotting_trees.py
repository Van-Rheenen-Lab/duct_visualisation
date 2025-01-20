import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def create_annotation_color_map(
        system_data: Dict[str, Any],
        fixed_annotation: Optional[str] = 'Endpoint',
        fixed_color: str = '#0080FE',  # Blue in HEX
        colormap_name: str = 'tab20',
) -> Dict[str, str]:
    """
    Creates a mapping from annotation label to hex color. If fixed_annotation is not None,
    we reserve a color for it and ensure that annotation has the given fixed_color.

    :param system_data: Dictionary containing segment data with annotations.
    :param fixed_annotation: (optional) The annotation that should keep a fixed color. If None, no fixed color is assigned.
    :param fixed_color: The HEX color for the fixed_annotation, if applicable.
    :param colormap_name: Name of the matplotlib colormap to use for assigning colors.
    :return: Dictionary of annotation -> color in hex format.
    """

    annotations = set()
    for seg_data in system_data.get('segments', {}).values():
        properties = seg_data.get('properties', {})
        annotation = properties.get('Annotation', [])

        # Ensure annotation is a list
        if isinstance(annotation, str):
            annotation = [annotation]
        elif not isinstance(annotation, list):
            # If 'Annotation' is not a string or list, skip
            continue

        annotations.update(annotation)

    # If we have a fixed_annotation, remove it from the dynamic set so it doesn't get a random color.
    if fixed_annotation:
        annotations.discard(fixed_annotation)

    cmap = plt.cm.get_cmap(colormap_name)
    num_colors = cmap.N

    annotation_to_color = {}
    sorted_annotations = sorted(annotations)  # Sort for consistent color assignment

    for i, annotation in enumerate(sorted_annotations):
        color = cmap((i + 2) % num_colors)
        # Convert RGBA to HEX
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(255 * color[0]),
            int(255 * color[1]),
            int(255 * color[2])
        )
        annotation_to_color[annotation] = color_hex

    # Only assign a color if fixed_annotation is not None
    if fixed_annotation:
        annotation_to_color[fixed_annotation] = fixed_color

    return annotation_to_color


def get_segment_color(segment_data, annotation_to_color, segment_color_map=None):
    """
    Get the color for a given segment. If segment_color_map is provided and the segment name
    is found there, use it. Otherwise, fall back to annotation_to_color.
    """

    segment_name = segment_data.get('segment_name')
    if segment_color_map and segment_name in segment_color_map:
        return segment_color_map[segment_name]

    if 'properties' in segment_data and 'Annotation' in segment_data['properties']:
        ann = segment_data['properties']['Annotation']
        return annotation_to_color.get(ann, 'red')

    return 'black'

def hierarchy_pos(G, root=None, vert_gap=0.2):
    if root is None:
        root = list(G.nodes)[0]
    pos = {}
    next_x = [-1]

    def recurse(node, depth, parent=None):
        children = list(G.neighbors(node))
        if parent is not None and parent in children:
            children.remove(parent)
        if len(children) == 0:
            pos[node] = (next_x[0], - depth * vert_gap)
            next_x[0] += 1
        else:
            child_x = []
            for child in children:
                recurse(child, depth + 1, node)
                child_x.append(pos[child][0])
            pos[node] = ((min(child_x) + max(child_x)) / 2, - depth * vert_gap)

    recurse(root, 0)
    return pos

def plot_hierarchical_graph(G, system_data=None, root_node=None,
                            use_hierarchy_pos=False, vert_gap=1,
                            orthogonal_edges=False, vert_length=1,
                            annotation_to_color=None, segment_color_map=None, linewidth = 1.5):
    """
    Revised function that can color segments by annotation or by an externally provided segment_color_map.
    """

    if not G.nodes:
        raise ValueError("The graph is empty.")

    # Determine the root node if not provided
    if root_node is None:
        root_node = list(G.nodes)[0]

    # Layout selection
    if use_hierarchy_pos:
        pos = hierarchy_pos(G, root=root_node, vert_gap=vert_gap)
    else:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB', root=root_node)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw edges
    for (u, v) in G.edges():
        segment_name = G[u][v].get('segment_name', None)
        c = 'black'
        segment_data = None
        if segment_name and system_data and 'segments' in system_data:
            seg_data = system_data['segments'].get(segment_name, {})
            # Add 'segment_name' to segment_data for convenience
            seg_data['segment_name'] = segment_name
            c = get_segment_color(seg_data, annotation_to_color, segment_color_map)

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if orthogonal_edges:
            # Ensure top-down orientation (parent above child)
            if y1 < y2:
                # Swap so we always draw top-down
                u, v = v, u
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Adjust y2 if node is a leaf
            if len(list(G.neighbors(v))) == 1:
                y2 = y2 - (vert_length - 1)

            # L-shaped edges
            ax.plot([x1, x2], [y1, y1], color=c, linewidth=linewidth, zorder=1)
            ax.plot([x2, x2], [y1, y2], color=c, linewidth=linewidth, zorder=1)
        else:
            # Straight edges
            ax.plot([x1, x2], [y1, y2], color=c, linewidth=linewidth, zorder=1)


    # If system_data and annotation_to_color are available, create a legend for annotations
    if system_data and annotation_to_color and not segment_color_map:
        legend_handles = []
        for ann, color in annotation_to_color.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color, label=ann))
        ax.legend(handles=legend_handles, title='Annotations', loc='lower center', bbox_to_anchor=(0.5, -1.2))

    # add a depth-level scale bar to the right
    lowest_y = min(y for x, y in pos.values())
    highest_y = max(y for x, y in pos.values())
    depth_range = highest_y - lowest_y
    depth_levels = int(depth_range / vert_gap)
    for i in range(depth_levels + 1):
        if i % 5 == 0:
            y = highest_y - i * vert_gap
            ax.text(-6, y, f" {i}", horizontalalignment='right', verticalalignment='center', fontsize=12)
            ax.plot([-5, -4], [y, y], color='black', linewidth=2)
        else:
            y = highest_y - i * vert_gap
            ax.plot([-5, -4], [y, y], color='black', linewidth=1)

    return fig, ax