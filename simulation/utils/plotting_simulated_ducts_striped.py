import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from collections import Counter


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

    For each edge, the cell (clone) list (from the attribute given by `clone_attr`)
    is partitioned into subsegments. If there are fewer clones than `subsegments`,
    the number of subsegments equals the number of clones (or 1 if none).
    For each subsegment, the majority annotation is determined and used to look up a color
    (via annotation_to_color) for drawing that stripe.
    """
    # Determine node positions.
    if use_hierarchy_pos:
        pos = hierarchy_pos(G, root=root_node, vert_gap=vert_gap)
    else:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB', root=root_node)

    fig, ax = plt.subplots(figsize=(35, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    for (u, v) in G.edges():
        # Retrieve node positions for the edge endpoints.
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Get the clone list for this edge.
        clones = G[u][v].get(clone_attr, [])
        M = len(clones)
        # Determine the number of subsegments.
        n_seg = subsegments if M >= subsegments else (M if M > 0 else 1)

        # Partition the clone list evenly and determine the majority annotation in each segment.
        clone_annotations = []
        if M > 0:
            for i in range(n_seg):
                start_idx = math.floor(i * M / n_seg)
                end_idx = math.floor((i + 1) * M / n_seg)
                subclone_list = clones[start_idx:end_idx]
                if subclone_list:
                    if annotation_to_color is not None:
                        # Compare as strings so that keys match.
                        annotated = [str(c) for c in subclone_list if str(c) in annotation_to_color]
                    else:
                        annotated = [str(c) for c in subclone_list]
                    if annotated:
                        cnt = Counter(annotated)
                        ann, _ = cnt.most_common(1)[0]
                        clone_annotations.append(ann)
                    else:
                        clone_annotations.append(None)
                else:
                    clone_annotations.append(None)
        else:
            clone_annotations = [None] * n_seg

        # Draw the subsegments.
        if orthogonal_edges:
            # Compute horizontal and vertical lengths.
            horiz_len = abs(x2 - x1)
            vert_len = abs(y2 - y1)
            total_len = horiz_len + vert_len

            # Flip nodes if necessary so that the horizontal part comes first.
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                horiz_len = abs(x2 - x1)
                vert_len = abs(y2 - y1)
                total_len = horiz_len + vert_len

            for i in range(n_seg):
                start_frac = i / n_seg
                end_frac = (i + 1) / n_seg
                dist_start = start_frac * total_len
                dist_end = end_frac * total_len

                # Choose a color based on the determined annotation.
                ann = clone_annotations[i]
                ann_str = str(ann) if ann is not None else None
                color = annotation_to_color.get(ann_str, 'black') if (annotation_to_color and ann_str) else 'black'

                # Case 1: Entirely in the horizontal segment.
                if dist_end <= horiz_len:
                    sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                    ex = x1 + (x2 - x1) * (dist_end / horiz_len) if horiz_len > 0 else x1
                    sy = y1
                    ey = y1
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                # Case 2: Entirely in the vertical segment.
                elif dist_start >= horiz_len:
                    vs = (dist_start - horiz_len) / vert_len if vert_len > 0 else 0
                    ve = (dist_end - horiz_len) / vert_len if vert_len > 0 else 0
                    sx = x2
                    ex = x2
                    sy = y1 + np.sign(y2 - y1) * (vert_len * vs)
                    ey = y1 + np.sign(y2 - y1) * (vert_len * ve)
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                # Case 3: The subsegment spans both horizontal and vertical parts.
                else:
                    # Horizontal part: from dist_start to end of horizontal section.
                    sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                    ex = x2
                    sy = y1
                    ey = y1
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                    # Vertical part: remainder of the subsegment.
                    ve = (dist_end - horiz_len) / vert_len if vert_len > 0 else 0
                    sx2 = x2
                    ex2 = x2
                    sy2 = y1
                    ey2 = y1 + np.sign(y2 - y1) * (vert_len * ve)
                    ax.plot([sx2, ex2], [sy2, ey2], color=color, linewidth=linewidth, zorder=1)
        else:
            # For non-orthogonal drawing, simply interpolate along the straight line.
            for i in range(n_seg):
                start_frac = i / n_seg
                end_frac = (i + 1) / n_seg
                sx = x1 + (x2 - x1) * start_frac
                sy = y1 + (y2 - y1) * start_frac
                ex = x1 + (x2 - x1) * end_frac
                ey = y1 + (y2 - y1) * end_frac
                ann = clone_annotations[i]
                color = annotation_to_color.get(str(ann), 'black') if (
                            annotation_to_color and ann is not None) else 'black'
                ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)

    # Optionally draw node labels.
    if draw_nodes:
        for node in G.nodes():
            x, y = pos[node]
            ax.text(x, y, str(node), fontsize=5, ha='center', va='center', zorder=3, color='navy')

    plt.tight_layout()
    return fig, ax
