import os
import math
import imageio
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from utils.loading_saving import load_duct_systems, create_duct_graph, select_biggest_duct_system
from utils.fixing_annotations import connect_component_to_main
from utils.plotting_trees import get_segment_color

def merge_disconnected_components(G, coord_tol=10):
    while True:
        components = list(nx.connected_components(G.to_undirected()))
        if len(components) <= 1:
            break
        merged_any = False
        main_comp = max(components, key=len)
        for comp in components:
            if comp == main_comp:
                continue
            if connect_component_to_main(G, main_comp, comp, coord_tol=coord_tol):
                merged_any = True
        if not merged_any:
            break
    return G

def plot_duct_network(ax, G, annotation_to_color, linewidth=1):
    ax.clear()
    # Edges
    for u, v, data in G.edges(data=True):
        x = [G.nodes[u]['x'], G.nodes[v]['x']]
        y = [G.nodes[u]['y'], G.nodes[v]['y']]
        z = [G.nodes[u].get('z', 0), G.nodes[v].get('z', 0)]
        c = get_segment_color(data, annotation_to_color)
        ax.plot(x, y, z, color=c, linewidth=linewidth)

    # Nodes in black
    xs, ys, zs = zip(*[(d['x'], d['y'], d.get('z', 0)) for n, d in G.nodes(data=True)])
    # ax.scatter(xs, ys, zs, color='black', s=10)

    ax.set_axis_off()
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

def create_animation(G, annotation_to_color, rotations=5, frames_per_rotation=72, output_gif="duct_network_rotation.gif"):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        angle = (360 / frames_per_rotation) * frame
        ax.view_init(elev=20, azim=angle)
        plot_duct_network(ax, G, annotation_to_color)

    total_frames = rotations * frames_per_rotation
    ani = FuncAnimation(fig, update, frames=total_frames, interval=50)

    # Save as GIF directly
    ani.save(output_gif, writer='pillow', fps=20, dpi=300)
    plt.close(fig)
    print(f"Animation saved as {output_gif}")

def main():
    # file_path = r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\Coseq_fixed.json"
    file_path = r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\A15_use-bp298-as-origin.json"
    # annotation_to_color = {"6": "#39FF14", "8": "#f528f2", "11": "#FFA500", "15": "#FF0000", "Endpoint": "#3689ff"}
    annotation_to_color = {"DCIS": "#FF0000", "Lesion": "#FF0000", "TDLU": "#0080FE"}

    duct_systems = load_duct_systems(file_path)
    system_data = select_biggest_duct_system(duct_systems)
    G = create_duct_graph(system_data)

    if len(G.nodes) == 0:
        print("Graph is empty for", file_path)
        return

    G = merge_disconnected_components(G)

    create_animation(G, annotation_to_color, rotations=1, frames_per_rotation=360)

if __name__ == "__main__":
    main()
