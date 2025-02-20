import os
import random
import json
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from puberty import simulate_ductal_tree
from adulthood import simulate_adulthood
from duct_excision_simulations import gather_clone_fractions_for_selected_ducts
from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph, find_root, select_biggest_duct_system
from analysis.utils.plotting_striped_trees import plot_hierarchical_graph_subsegments
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from rasterio.features import rasterize
from skimage import io
from plotting_simulated_ducts_striped import plot_hierarchical_graph_subsegments_simulated
from plotting_simulated_ducts import plot_selected_ducts

def plot_branch_level_distribution(G, title="Branch Level Distribution", save_path=None):
    """
    Computes the shortest-path levels from a root (a node with in-degree 0)
    and plots a bar chart of the number of ducts (nodes) per level.
    """
    root = find_root(G)
    level_dict = nx.shortest_path_length(G, source=root)
    counts = {}
    for node, lvl in level_dict.items():
        counts[lvl] = counts.get(lvl, 0) + 1
    levels_sorted = sorted(counts.items())
    x = [lvl for lvl, cnt in levels_sorted]
    y = [cnt for lvl, cnt in levels_sorted]

    # Use large figure size
    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Branch Level")
    plt.ylabel("Number of Ducts")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

def save_figure_with_separated_legend(fig, base_filename, dpi=900, output_folder="output_images"):
    """
    Ensures the figure is set to figsize (35,12), then checks each axis for a legend.
    If found, it extracts the handles, labels, and title, removes the legend from the axes,
    saves the main figure (without the legend)
    """
    # Force the figure to be large
    fig.set_size_inches(35, 12)

    # Search for a legend in the axes
    legend_obj = None
    target_ax = None
    for ax in fig.get_axes():
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            target_ax = ax
            break

    main_path = os.path.join(output_folder, base_filename + ".png")

    if legend_obj is not None:
        # Extract legend data
        handles, labels = target_ax.get_legend_handles_labels()
        # If no legend items, simply save the figure and return.
        if not handles or not labels:
            fig.savefig(main_path, dpi=dpi, bbox_inches='tight')
            return
        # Get legend title if available
        leg_title = ""
        if legend_obj.get_title() is not None:
            leg_title = legend_obj.get_title().get_text()
        # Remove the legend from the axis
        legend_obj.remove()
        # Save the main figure without the legend
        fig.savefig(main_path, dpi=dpi, bbox_inches='tight')

        # Create a new figure solely for the legend.
        figLegend = plt.figure(figsize=(35, 12))
        dummy_ax = figLegend.add_subplot(111)
        dummy_ax.axis('off')
        # Re-create the legend using the handles and labels.
        legend = dummy_ax.legend(
            handles,
            labels,
            title=leg_title,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(labels) if len(labels) > 0 else 1
        )
        legend.get_frame().set_linewidth(0)
        legend_path = os.path.join(output_folder, base_filename + "_legend.png")
        figLegend.savefig(legend_path, dpi=dpi, bbox_inches='tight')
        plt.close(figLegend)
    else:
        fig.savefig(main_path, dpi=dpi, bbox_inches='tight')


def main():
    random.seed(42)

    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    # --- Define the clone color map (for highlighting simulation clones) ---
    clone_color_map = {"84": "#ff7f0e", "58": "#2ca02c", "21": "#b02797"}
    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 300

    n_clones = 170
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    max_cells = 6_000_000
    subsegments = 50

    vert_gap = 3.5

    # Load real duct system(s)
    duct_systems = load_duct_systems(json_path)

    # Select the biggest duct system
    duct_system = select_biggest_duct_system(duct_systems)


    # Create duct mask from borders
    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)
    valid_geoms = []
    for feat in duct_borders['features']:
        geom = shape(feat['geometry'])
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_valid:
            valid_geoms.append(geom)
    duct_polygon = unary_union(valid_geoms)

    # Get image shape from red image (if available)
    if os.path.exists(red_image_path):
        red_image = io.imread(red_image_path)
        base_shape = red_image.shape
    else:
        red_image = None
        base_shape = (1024, 1024)  # default fallback

    green_image = io.imread(green_image_path) if green_image_path else None
    yellow_image = io.imread(yellow_image_path) if yellow_image_path else None

    shapes = [(duct_polygon, 1)]
    duct_mask = rasterize(shapes, out_shape=base_shape, fill=0, dtype=np.uint8)

    # Build directed graph for real data
    G_real = create_directed_duct_graph(duct_system)

    # Determine a good root for the BFS (using the first branch point)
    first_bp = list(G_real.nodes())[0]
    while len(list(G_real.predecessors(first_bp))) == 1:
        first_bp = list(G_real.predecessors(first_bp))[0]
    root_node = first_bp
    print(f"Using branch point {root_node} as BFS root")

    # Plot the striped tree for the real network
    fig_real, ax_real = plot_hierarchical_graph_subsegments(
        G=G_real,
        root_node=root_node,
        duct_mask=duct_mask,
        red_image=red_image,
        green_image=green_image,
        yellow_image=yellow_image,
        draw_nodes=False,
        threshold=threshold_value,
        N=subsegments,
        use_hierarchy_pos=True,
        vert_gap=vert_gap,
        orthogonal_edges=True,
        linewidth=0.6,
        buffer_width=10
    )
    plt.title("Real Network: Striped Tree")
    fig_real.set_size_inches(35, 12)
    fig_real.savefig(os.path.join(output_folder, "real_network_striped_tree.png"), dpi=600, bbox_inches='tight')
    plt.close(fig_real)

    # Plot branch-level distribution for the real network
    plot_branch_level_distribution(
        G_real,
        title="Real Network: Ducts per Branch Level",
        save_path=os.path.join(output_folder, "real_network_branch_level_distribution.png")
    )
    plt.close()

    # -- Simulate Puberty --
    G_puberty, progress_data_puberty = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob
    )
    # For simulated trees, select a root node (here we use the first node)
    root_node_sim = list(G_puberty.nodes())[0]

    # Plot the puberty duct tree using the new sub-segmented plotting function.
    fig_puberty, ax_puberty = plot_hierarchical_graph_subsegments_simulated(
        G_puberty,
        root_node=root_node_sim,
        clone_attr="duct_clones",
        annotation_to_color=clone_color_map,
        subsegments=subsegments,
        use_hierarchy_pos=True,
        vert_gap=vert_gap,
        orthogonal_edges=True,
        linewidth=0.9,
        draw_nodes=False
    )
    plt.title("Simulated Ductal Tree after Puberty (Subsegmented)")
    fig_puberty.set_size_inches(35, 12)
    save_figure_with_separated_legend(fig_puberty, "puberty_duct_tree", dpi=600, output_folder=output_folder)
    plt.close(fig_puberty)

    # Plot branch-level distribution for the puberty simulation
    plot_branch_level_distribution(
        G_puberty,
        title="Simulation (Puberty): Ducts per Branch Level",
        save_path=os.path.join(output_folder, "puberty_branch_level_distribution.png")
    )
    plt.close()

    # Determine selected ducts (for heatmap analysis)
    nodes_puberty = list(G_puberty.nodes())
    if nodes_puberty:
        last_bp = nodes_puberty[-1]
        selected_ducts = [last_bp]
        while True:
            parents = list(G_puberty.predecessors(selected_ducts[0]))
            if not parents:
                break
            selected_ducts = parents + selected_ducts
        # Skip the first few and take every 3rd duct:
        selected_ducts = selected_ducts[5:]
        selected_ducts = selected_ducts[::3]
    else:
        selected_ducts = []

    plot_selected_ducts(G_puberty, selected_ducts, vert_gap=vert_gap)
    plt.title("Simulated Puberty: Selected Ducts for Heatmap Analysis")
    fig_selected = plt.gcf()
    fig_selected.set_size_inches(35, 12)
    # turn off the legend for this plot
    fig_selected.legend().remove()
    save_figure_with_separated_legend(fig_selected, "puberty_selected_ducts", dpi=600, output_folder=output_folder)
    plt.close(fig_selected)

    # -- Simulate Adulthood --
    G_adulthood, progress_data_adult = simulate_adulthood(G_puberty.copy(), rounds=33)
    root_node_sim_adult = list(G_adulthood.nodes())[0]

    # Plot the adulthood duct tree (using pubertal clone info, i.e. clone_attr "duct_clones")
    fig_adult, ax_adult = plot_hierarchical_graph_subsegments_simulated(
        G_adulthood,
        root_node=root_node_sim_adult,
        clone_attr="duct_clones",
        annotation_to_color=clone_color_map,
        subsegments=subsegments,
        use_hierarchy_pos=True,
        vert_gap=vert_gap,
        orthogonal_edges=True,
        linewidth=0.9,
        draw_nodes=False
    )
    plt.title("Simulated Ductal Tree after Adulthood (Subsegmented, Pubertal Clones)")
    fig_adult.set_size_inches(35, 12)
    save_figure_with_separated_legend(fig_adult, "adulthood_duct_tree_highlighted", dpi=900, output_folder=output_folder)
    plt.close(fig_adult)

    # Create a new clone_color_map for adult clones
    adult_clones = []
    for u, v in G_adulthood.edges():
        adult_clones.extend(G_adulthood[u][v].get("adult_clones", []))
    adult_clones = list(set(adult_clones))
    # random_adult_clones = ["A_63072", "A_225115", "A_167763", "A_166034", "A_24769",
    #                        "A_165140", "A_270905", "A_217109", "A_47697", "A_89327", "A_9988"]

    # Make a random selection of 1000 adult clones
    random_adult_clones = random.sample(adult_clones, 1000)

    # make color map
    clone_color_map_adult = {str(c): sns.color_palette("tab20")[i % 20] for i, c in enumerate(random_adult_clones)}


    # Plot the adulthood duct tree with adult clones highlighted (using clone_attr "adult_clones")
    fig_adult2, ax_adult2 = plot_hierarchical_graph_subsegments_simulated(
        G_adulthood,
        root_node=root_node_sim_adult,
        clone_attr="adult_clones",
        annotation_to_color=clone_color_map_adult,
        subsegments=subsegments,
        use_hierarchy_pos=True,
        vert_gap=vert_gap,
        orthogonal_edges=True,
        linewidth=0.9,
        draw_nodes=False
    )
    plt.title("Simulated Ductal Tree after Adulthood (Subsegmented, Adult Clones)")
    fig_adult2.set_size_inches(35, 12)
    save_figure_with_separated_legend(fig_adult2, "adulthood_duct_tree_adult_clones", dpi=900, output_folder=output_folder)
    plt.close(fig_adult2)

    df_puberty = gather_clone_fractions_for_selected_ducts(
        G_puberty, selected_ducts, puberty_clones=True
    )
    df_adulthood_pubertal = gather_clone_fractions_for_selected_ducts(
        G_adulthood, selected_ducts, puberty_clones=True
    )
    df_adulthood_adult = gather_clone_fractions_for_selected_ducts(
        G_adulthood, selected_ducts, puberty_clones=False, restrict_to_selected=True
    )

    # Cluster sorting for heatmaps
    vaf_threshold = 0
    vaf_columns = selected_ducts  # these are the columns used to compute the pattern
    num_bits = len(vaf_columns)
    pattern_order = []
    for r in range(1, num_bits + 1):
        for combo in itertools.combinations(range(num_bits), r):
            pattern = ['0'] * num_bits
            for bit_index in combo:
                pattern[bit_index] = '1'
            pattern_order.append("".join(pattern))
    pattern_to_cluster = {p: i for i, p in enumerate(pattern_order)}

    def assign_pattern(row, threshold=vaf_threshold, columns=vaf_columns):
        bits = []
        for c in columns:
            bits.append('1' if row[c] > threshold else '0')
        return "".join(bits)

    # For the puberty (pubertal clones) dataframe:
    df_puberty['Pattern'] = df_puberty.apply(assign_pattern, axis=1)
    df_puberty['Cluster'] = df_puberty['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))
    df_puberty.sort_values('Cluster', inplace=True)

    # Remove the fully negative cluster (i.e. all zeros) from df_puberty:
    nonzero_pattern = '0' * len(vaf_columns)
    df_puberty = df_puberty[df_puberty['Pattern'] != nonzero_pattern]

    # Now reindex the pubertal adulthood dataframe to follow the ordering of df_puberty:
    df_adulthood_pubertal = df_adulthood_pubertal.reindex(df_puberty.index, fill_value=0.0)

    # For the adult clones dataframe:
    df_adulthood_adult['Pattern'] = df_adulthood_adult.apply(assign_pattern, axis=1)
    df_adulthood_adult['Cluster'] = df_adulthood_adult['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))
    df_adulthood_adult.sort_values('Cluster', inplace=True)
    # Remove the fully negative cluster here as well:
    df_adulthood_adult = df_adulthood_adult[df_adulthood_adult['Pattern'] != nonzero_pattern]

    # Plot heatmap for puberty clone fractions
    sns.heatmap(df_puberty[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clone IDs")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Puberty (Clone Fractions)")
    plt.savefig(os.path.join(output_folder, "heatmap_puberty_clone_fractions.png"), dpi=900, bbox_inches='tight')
    plt.close()

    # Plot heatmap for adulthood (pubertal clones)
    sns.heatmap(df_adulthood_pubertal[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clone IDs (Pubertal)")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Adulthood (Pubertal Clones)")
    plt.savefig(os.path.join(output_folder, "heatmap_adulthood_pubertal_clone_fractions.png"), dpi=900,
                bbox_inches='tight')
    plt.close()

    # Plot heatmap for adulthood (adult clones)
    sns.heatmap(df_adulthood_adult[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clone IDs (Adult)")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Adulthood (Adult Clones)")
    plt.savefig(os.path.join(output_folder, "heatmap_adulthood_adult_clone_fractions.png"), dpi=900,
                bbox_inches='tight')
    plt.close()

    plt.show()


if __name__ == "__main__":
    main()
