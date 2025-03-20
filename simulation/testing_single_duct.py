import copy
import random
import matplotlib.pyplot as plt
import colorcet as cc  # for Glasbey palette
import numpy as np
import imageio
import networkx as nx
import math
from collections import Counter

# Import your simulation functions (assumed to be available)
from puberty import simulate_ductal_tree
from adulthood import simulate_adulthood

def extract_unique_clone_order(clone_list):
    unique = []
    seen = set()
    for cid in clone_list:
        if cid not in seen:
            unique.append(cid)
            seen.add(cid)
    return unique

# --- Dendrogram helper functions from your provided script ---

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

def plot_hierarchical_graph_subsegments_simulated(
        G,
        root_node,
        clone_attr="duct_clones",
        annotation_to_color=None,
        use_hierarchy_pos=True,
        vert_gap=0.5,
        orthogonal_edges=True,
        linewidth=1.5,
        draw_nodes=False
):
    # Determine node positions.
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

        # Get the clone list for this edge.
        clones = G[u][v].get(clone_attr, [])
        M = len(clones)
        # Use exactly one subsegment per cell in the duct.
        n_seg = M if M > 0 else 1

        # Partition the clone list into subsegments and determine the majority annotation.
        # In this case each segment represents one cell.
        clone_annotations = []
        if M > 0:
            for i in range(n_seg):
                # For one cell per segment, each segment is just that one cell.
                subclone_list = [clones[i]]
                ann = str(subclone_list[0]) if subclone_list[0] is not None else None
                clone_annotations.append(ann)
        else:
            clone_annotations = [None]

        # Draw the subsegments.
        if orthogonal_edges:
            horiz_len = abs(x2 - x1)
            vert_len = abs(y2 - y1)
            total_len = horiz_len + vert_len

            # Ensure horizontal part comes first.
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

                ann = clone_annotations[i]
                ann_str = str(ann) if ann is not None else None
                color = annotation_to_color.get(ann_str, 'black') if (annotation_to_color and ann_str) else 'black'

                if dist_end <= horiz_len:
                    sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                    ex = x1 + (x2 - x1) * (dist_end / horiz_len) if horiz_len > 0 else x1
                    sy = y1
                    ey = y1
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                elif dist_start >= horiz_len:
                    vs = (dist_start - horiz_len) / vert_len if vert_len > 0 else 0
                    ve = (dist_end - horiz_len) / vert_len if vert_len > 0 else 0
                    sx = x2
                    ex = x2
                    sy = y1 + np.sign(y2 - y1) * (vert_len * vs)
                    ey = y1 + np.sign(y2 - y1) * (vert_len * ve)
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                else:
                    # Spanning horizontal and vertical segments.
                    sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                    ex = x2
                    sy = y1
                    ey = y1
                    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                    ve = (dist_end - horiz_len) / vert_len if vert_len > 0 else 0
                    sx2 = x2
                    ex2 = x2
                    sy2 = y1
                    ey2 = y1 + np.sign(y2 - y1) * (vert_len * ve)
                    ax.plot([sx2, ex2], [sy2, ey2], color=color, linewidth=linewidth, zorder=1)
        else:
            # Straight line interpolation if not using orthogonal edges.
            for i in range(n_seg):
                start_frac = i / n_seg
                end_frac = (i + 1) / n_seg
                sx = x1 + (x2 - x1) * start_frac
                sy = y1 + (y2 - y1) * start_frac
                ex = x1 + (x2 - x1) * end_frac
                ey = y1 + (y2 - y1) * end_frac
                ann = clone_annotations[i]
                color = annotation_to_color.get(str(ann), 'black') if (annotation_to_color and ann is not None) else 'black'
                ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)

    if draw_nodes:
        for node in G.nodes():
            x, y = pos[node]
            ax.text(x, y, str(node), fontsize=8, ha='center', va='center', zorder=3, color='navy')

    plt.tight_layout()
    return fig, ax

# --- Helper to extract a subtree (duct 249 plus children and grandchildren) ---

def extract_subtree(G, root, max_depth):
    nodes = set()
    queue = [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        nodes.add(node)
        for child in G.successors(node):
            if child not in nodes:
                queue.append((child, depth + 1))
    return G.subgraph(nodes).copy()

# --- Main routine ---

def main():
    random.seed(42)
    max_cells = 6_000_000
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    initial_side_count = 85
    initial_center_count = 85
    duct_id = 249  # starting duct for the subtree
    n_iterations = 33  # number of iterations for adult simulation (and the GIF)

    print("Simulating puberty...")
    # Run the puberty simulation.
    G_puberty, _ = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=initial_side_count,
        initial_center_count=initial_center_count,
        initial_termination_prob=initial_termination_prob
    )
    # Preserve the original pubertal state.
    G_puberty_orig = copy.deepcopy(G_puberty)

    # --- Create static dendrogram from the original pubertal graph ---
    subtree_orig = extract_subtree(G_puberty_orig, duct_id, max_depth=2)
    # Build a unified color map for pubertal clones from the original subtree.
    union_clones = []
    for u, v, data in subtree_orig.edges(data=True):
        union_clones.extend(data.get("duct_clones", []))
    unique_clones = extract_unique_clone_order(union_clones)
    annotation_to_color = {str(cid): cc.glasbey[i] for i, cid in enumerate(unique_clones)}

    # Plot the static dendrogram.
    fig_static, ax_static = plot_hierarchical_graph_subsegments_simulated(
        subtree_orig,
        root_node=duct_id,
        clone_attr="duct_clones",
        annotation_to_color=annotation_to_color,
        use_hierarchy_pos=True,
        vert_gap=0.5,
        orthogonal_edges=True,
        linewidth=1.5,
        draw_nodes=True
    )
    output_static = "subtree_dendrogram.png"
    plt.savefig(output_static, dpi=300, bbox_inches='tight')
    print(f"Static dendrogram saved to {output_static}")
    plt.close(fig_static)

    # --- Generate animated GIF for the dendrogram evolution ---
    gif_frames = []
    for i in range(n_iterations + 1):
        print(f"Simulating iteration {i} for GIF...")
        # Run adult simulation for i rounds.
        G_iter = simulate_adulthood(copy.deepcopy(G_puberty_orig), rounds=i)[0]
        # Extract subtree from the updated graph.
        subtree_iter = extract_subtree(G_iter, duct_id, max_depth=2)
        # Plot the dendrogram for this iteration.
        fig_iter, ax_iter = plot_hierarchical_graph_subsegments_simulated(
            subtree_iter,
            root_node=duct_id,
            clone_attr="duct_clones",
            annotation_to_color=annotation_to_color,
            use_hierarchy_pos=True,
            vert_gap=0.5,
            orthogonal_edges=True,
            linewidth=15,
            draw_nodes=True
        )
        ax_iter.set_title(f"Pubertal Clones Dendrogram at Iteration {i}", fontsize=20)
        fig_iter.canvas.draw()
        img = np.frombuffer(fig_iter.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig_iter.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(img)
        plt.close(fig_iter)
    output_gif = "subtree_dendrogram_evolution.gif"
    imageio.mimsave(output_gif, gif_frames, duration=1.5)
    print(f"Animated GIF saved to {output_gif}")

if __name__ == "__main__":
    main()
