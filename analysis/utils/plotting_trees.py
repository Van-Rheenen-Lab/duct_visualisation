import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple


def create_annotation_color_map(
        G: nx.Graph,
        fixed_annotation: Optional[str] = 'Endpoint',
        fixed_color: str = '#0080FE',  # Blue in HEX
        colormap_name: str = 'tab20'
) -> Dict[str, str]:
    """
    Creates a mapping from annotation labels to HEX colors using the graph's edge metadata.
    If fixed_annotation is provided, it reserves that annotation's color.
    """
    annotations = set()
    for _, _, data in G.edges(data=True):
        properties = data.get('properties', {})
        ann = properties.get('Annotation', [])
        if isinstance(ann, str):
            ann = [ann]
        elif not isinstance(ann, list):
            continue
        annotations.update(ann)

    if fixed_annotation:
        annotations.discard(fixed_annotation)

    cmap = plt.cm.get_cmap(colormap_name)
    num_colors = cmap.N

    annotation_to_color = {}
    try:
        sorted_annotations = sorted(annotations, key=lambda x: int(x))
    except Exception:
        sorted_annotations = sorted(annotations)

    for i, annotation in enumerate(sorted_annotations):
        color = cmap((i + 2) % num_colors)
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(255 * color[0]),
            int(255 * color[1]),
            int(255 * color[2])
        )
        annotation_to_color[annotation] = color_hex

    if fixed_annotation:
        annotation_to_color[fixed_annotation] = fixed_color

    return annotation_to_color


def get_segment_color(
        segment_data: Dict[str, Any],
        annotation_to_color: Dict[str, str],
        segment_color_map: Optional[Dict[str, str]] = None,
        is_endpoint: bool = False
) -> str:
    """
    Determine the color for a segment based on its metadata.
    Checks an override mapping (segment_color_map) first, then falls back to the annotation color.

    If an annotation is provided:
      - If it exists in the mapping, its color is returned.
      - Otherwise, blue (#3689ff) is returned if the segment is an endpoint, or black if not.

    If no annotation is provided, always return black.
    """
    segment_name = segment_data.get('segment_name', None)
    if segment_color_map and segment_name in segment_color_map:
        return segment_color_map[segment_name]

    properties = segment_data.get('properties', {})
    ann = properties.get('Annotation', None)
    if isinstance(ann, list) and ann:
        ann = ann[0]  # Use the first annotation if multiple are provided.

    if ann:
        return annotation_to_color.get(ann, '#3689ff' if is_endpoint else 'black')
    else:
        # No annotation provided: always return black.
        return 'black'


def hierarchy_pos(G: nx.Graph, root: Optional[str] = None, vert_gap: float = 0.2) -> Dict[Any, Tuple[float, float]]:
    """
    Compute a hierarchical layout for a graph.
    """
    if root is None:
        root = list(G.nodes)[0]
    pos = {}
    next_x = [-1]

    def recurse(node, depth, parent=None):
        children = list(G.neighbors(node))
        if parent is not None and parent in children:
            children.remove(parent)
        if not children:
            pos[node] = (next_x[0], -depth * vert_gap)
            next_x[0] += 1
        else:
            child_x = []
            for child in children:
                recurse(child, depth + 1, node)
                child_x.append(pos[child][0])
            pos[node] = ((min(child_x) + max(child_x)) / 2, -depth * vert_gap)

    recurse(root, 0)
    return pos


def plot_hierarchical_graph(
        G: nx.Graph,
        root_node: Optional[str] = None,
        use_hierarchy_pos: bool = False,
        vert_gap: float = 1,
        orthogonal_edges: bool = False,
        vert_length: float = 1,
        annotation_to_color: Optional[Dict[str, str]] = None,
        segment_color_map: Optional[Dict[str, str]] = None,
        linewidth: float = 1.5,
        legend_offset: float = -0.1
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a hierarchical graph where segment colors are determined by edge metadata.
    """
    if not G.nodes:
        raise ValueError("The graph is empty.")

    if root_node is None:
        root_node = list(G.nodes)[0]

    if use_hierarchy_pos:
        pos = hierarchy_pos(G, root=root_node, vert_gap=vert_gap)
    else:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB', root=root_node)

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # For each edge, determine if it goes to a leaf (child with no children)
    for u, v in G.edges():
        segment_data = G[u][v]
        # Determine positions.
        try:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
        except KeyError:
            print(f"Skipping edge {u} -> {v} due to missing node position.")
            continue

        # Determine which node is deeper (child) using y-coordinate (more negative is deeper).
        is_endpoint = False
        if y1 != y2:
            if y1 < y2:
                child = u
            else:
                child = v
            # In a tree, a leaf is typically a node with degree 1 (and not the root).
            if child != root_node and G.degree(child) == 1:
                is_endpoint = True

        # Get the segment color using our new is_endpoint flag.
        c = (get_segment_color(segment_data, annotation_to_color, segment_color_map, is_endpoint)
             if annotation_to_color else 'black')

        if orthogonal_edges:
            # Ensure top-down orientation.
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if len(list(G.neighbors(v))) == 1:
                y2 = y2 - (vert_length - 1)
            ax.plot([x1, x2], [y1, y1], color=c, linewidth=linewidth, zorder=1)
            ax.plot([x2, x2], [y1, y2], color=c, linewidth=linewidth, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color=c, linewidth=linewidth, zorder=1)

    if annotation_to_color and not segment_color_map:
        legend_handles = []
        for ann, color in annotation_to_color.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color, label=ann))
        ax.legend(handles=legend_handles, title='Annotations',
                  loc='lower center', bbox_to_anchor=(0.6, legend_offset))

    # Add a depth-level scale bar on the left.
    lowest_y = min(y for x, y in pos.values())
    highest_y = max(y for x, y in pos.values())
    depth_range = highest_y - lowest_y
    depth_levels = int(depth_range / vert_gap)
    for i in range(depth_levels + 1):
        y = highest_y - i * vert_gap
        if i % 5 == 0:
            ax.text(-6, y, f" {i}", horizontalalignment='right',
                    verticalalignment='center', fontsize=12)
            ax.plot([-5, -4], [y, y], color='black', linewidth=2)
        else:
            ax.plot([-5, -4], [y, y], color='black', linewidth=1)

    return fig, ax
