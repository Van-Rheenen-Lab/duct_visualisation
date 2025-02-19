import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
import math


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


def plot_hierarchical_graph_subsegments_simulated(
        G,
        root_node,
        clone_attr="duct_clones",
        annotation_to_color=None,
        subsegments=100,
        use_hierarchy_pos=True,
        vert_gap=1,
        orthogonal_edges=True,
        linewidth=1.5,
        draw_nodes=False
):
    """
    Plots a simulated ductal tree with each edge (duct) split into subsegments.

    For each edge, the cell (clone) list (given by `clone_attr`) is partitioned
    into subsegments. If there are fewer cells than desired_subsegments, the number
    of subsegments equals the number of cells (or 1 if there are none). For each
    subsegment, the majority annotation is used to look up a color (via
    annotation_to_color) to draw that stripe.
    """

    # Get node positions using hierarchy or graphviz layout.
    if use_hierarchy_pos:
        pos = hierarchy_pos(G, root=root_node, vert_gap=vert_gap)
    else:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB', root=root_node)

    fig, ax = plt.subplots(figsize=(35, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    for (u, v) in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Get the list of clones (simulated cells) for this duct.
        clones = G[u][v].get(clone_attr, [])
        M = len(clones)
        # If no clones, we still draw a single stripe.
        if M > 0:
            n_seg = subsegments if M >= subsegments else M
        else:
            n_seg = 1

        # Partition the clones evenly (assumes order corresponds to position)
        clone_annotations = []
        if M > 0:
            for i in range(n_seg):
                start_idx = math.floor(i * M / n_seg)
                end_idx = math.floor((i + 1) * M / n_seg)
                subclone_list = clones[start_idx:end_idx]
                if subclone_list:
                    # Only consider clones that are in the annotation color map.
                    if annotation_to_color is not None:
                        annotated_subclones = [c for c in subclone_list if str(c) in annotation_to_color]
                    else:
                        annotated_subclones = subclone_list
                    if annotated_subclones:
                        cnt = Counter(annotated_subclones)
                        ann, _ = cnt.most_common(1)[0]
                        clone_annotations.append(ann)
                    else:
                        clone_annotations.append(None)
                else:
                    clone_annotations.append(None)
        else:
            clone_annotations = [None] * n_seg

        if orthogonal_edges:
            # Determine lengths for an orthogonal route:
            horiz_len = abs(x2 - x1)
            vert_len = abs(y2 - y1)
            total_len = horiz_len + vert_len
            # If needed, flip nodes so the horizontal part comes first.
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                horiz_len = abs(x2 - x1)
                vert_len = abs(y2 - y1)
                total_len = horiz_len + vert_len

            # For each subsegment, compute its start and end distance along the duct.
            for i in range(n_seg):
                start_frac = i / n_seg
                end_frac = (i + 1) / n_seg
                dist_start = start_frac * total_len
                dist_end = end_frac * total_len

                # Choose color based on annotation (default to black if no mapping).
                ann = clone_annotations[i]
                ann_str = str(ann) if ann is not None else None
                if annotation_to_color and ann_str in annotation_to_color:
                    color = annotation_to_color[ann_str]
                else:
                    color = 'black'

                # Case 1: Entirely in horizontal part.
                if dist_end <= horiz_len:
                    sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                    ex = x1 + (x2 - x1) * (dist_end / horiz_len) if horiz_len > 0 else x1
                    sy = y1
                    ey = y1
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                # Case 2: Entirely in vertical part.
                elif dist_start >= horiz_len:
                    vs = (dist_start - horiz_len) / vert_len if vert_len > 0 else 0
                    ve = (dist_end - horiz_len) / vert_len if vert_len > 0 else 0
                    sx = x2
                    ex = x2
                    sy = y1 + np.sign(y2 - y1) * (vert_len * vs)
                    ey = y1 + np.sign(y2 - y1) * (vert_len * ve)
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                # Case 3: Subsegment spans horizontal and vertical parts.
                else:
                    # Horizontal part from dist_start to the end of horizontal segment.
                    sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                    ex = x2  # end of horizontal section
                    sy = y1
                    ey = y1
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                    # Vertical part for the remainder.
                    ve = (dist_end - horiz_len) / vert_len if vert_len > 0 else 0
                    sx2 = x2
                    ex2 = x2
                    sy2 = y1
                    ey2 = y1 + np.sign(y2 - y1) * (vert_len * ve)
                    ax.plot([sx2, ex2], [sy2, ey2], color=color, linewidth=linewidth, zorder=1)
        else:
            # Non-orthogonal: simply interpolate between (x1,y1) and (x2,y2).
            for i in range(n_seg):
                start_frac = i / n_seg
                end_frac = (i + 1) / n_seg
                sx = x1 + (x2 - x1) * start_frac
                sy = y1 + (y2 - y1) * start_frac
                ex = x1 + (x2 - x1) * end_frac
                ey = y1 + (y2 - y1) * end_frac
                ann = clone_annotations[i]
                if annotation_to_color and ann in annotation_to_color:
                    color = annotation_to_color[ann]
                else:
                    color = 'black'
                ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)

    if draw_nodes:
        for node in G.nodes():
            x, y = pos[node]
            ax.text(x, y, str(node), fontsize=5, ha='center', va='center', zorder=3, color='navy')

    plt.tight_layout()
    return fig, ax
