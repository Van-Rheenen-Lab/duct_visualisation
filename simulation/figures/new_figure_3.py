import os
import random
import json
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from simulation.puberty_deposit_elimination import simulate_ductal_tree
from simulation.adulthood import simulate_adulthood
from simulation.utils.duct_excision_simulations import gather_clone_fractions_for_selected_ducts
from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph, find_root, \
    select_biggest_duct_system
from analysis.utils.plotting_striped_trees import plot_hierarchical_graph_subsegments
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from rasterio.features import rasterize
from skimage import io
from simulation.utils.plotting_simulated_ducts_striped import plot_hierarchical_graph_subsegments_simulated
from simulation.utils.plotting_simulated_ducts import plot_selected_ducts
from analysis.utils.cumulative_area import compute_branch_levels

def plot_branch_level_distribution(G, title="Branch Level Distribution", save_path=None):
    """
    Computes the shortest-path levels from a root (a node with in-degree 0)
    and plots a bar chart of the number of ducts (nodes) per level.
    """
    root = find_root(G)

    levels = compute_branch_levels(G, root)
    counts = np.bincount(list(levels.values()))
    x = np.arange(counts.size)
    y = counts

    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Branch Level")
    plt.ylabel("Number of Ducts")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')


import numpy as np, collections as _c

def puberty_vs_adult0(G_puberty, progress_adult):
    # adult-0 pubertal counts
    ad_counts = progress_adult["pubertal_id_counts"][0]          # {pub_id: n_cells}
    ad_avg    = np.mean(list(ad_counts.values())) if ad_counts else 0.
    ad_total  = len(ad_counts)

    # puberty counts
    pub_counts = _c.Counter()                                    # {pub_id: n_cells}
    for _, _, e in G_puberty.edges(data=True):
        pub_counts.update(e.get("duct_clones", []))
    pub_avg   = np.mean(list(pub_counts.values())) if pub_counts else 0.
    pub_total = len(pub_counts)

    # clones lost after selection
    lost_ids  = set(pub_counts) - set(ad_counts)
    lost_avg  = np.mean([pub_counts[i] for i in lost_ids]) if lost_ids else 0.

    return dict(
        avg_puberty = pub_avg,
        avg_adult0  = ad_avg,
        delta_avg   = ad_avg - pub_avg,
        total_puberty = pub_total,
        total_adult0  = ad_total,
        retention   = ad_total / pub_total if pub_total else 0.,
        avg_cells_lost = lost_avg          # ⟨pubertal cells⟩ of eliminated clones
    )


def save_figure_with_separated_legend(fig, base_filename, dpi=900, output_folder="output_images"):
    """
    Forces the figure size to (35,12), then checks each axis for a legend.
    If found, it extracts the handles, labels, and title, removes the legend from the axes,
    saves the main figure (without the legend) and also saves a separate legend figure.
    """
    fig.set_size_inches(35, 12)
    legend_obj = None
    target_ax = None
    for ax in fig.get_axes():
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            target_ax = ax
            break

    main_path = os.path.join(output_folder, base_filename + ".png")
    if legend_obj is not None:
        handles, labels = target_ax.get_legend_handles_labels()
        if not handles or not labels:
            fig.savefig(main_path, dpi=dpi, bbox_inches='tight')
            return
        leg_title = ""
        if legend_obj.get_title() is not None:
            leg_title = legend_obj.get_title().get_text()
        legend_obj.remove()
        fig.savefig(main_path, dpi=dpi, bbox_inches='tight')
        figLegend = plt.figure(figsize=(35, 12))
        dummy_ax = figLegend.add_subplot(111)
        dummy_ax.axis('off')
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


def gather_clones_for_duct(G, duct, puberty_clones=True):
    """
    Given a graph G and a duct node ID, returns the list of clone IDs
    from the incoming edge to that duct.
    For pubertal clones, returns the "duct_clones" attribute.
    For adult clones, returns the "adult_clones" attribute.
    """
    parents = list(G.predecessors(duct))
    if not parents:
        return []
    parent = parents[0]
    if puberty_clones:
        return G[parent][duct].get("duct_clones", [])
    else:
        return G[parent][duct].get("adult_clones", [])


def plot_ductal_clone_line_at_y(clone_ids, ax, highlight_map, y=0, line_width=6, default_color="black"):
    """
    Plots a horizontal line for the provided clone IDs at vertical position y.
    Each segment represents one cell along the duct.
    For each clone, if its ID is in highlight_map then that color is used;
    otherwise, the color given by default_color is used.
    """
    n = len(clone_ids)
    if n == 0:
        return
    cell_width = 1.0 / n
    x_positions = [i * cell_width for i in range(n + 1)]
    for i in range(n - 1):
        cid = str(clone_ids[i])
        color = highlight_map[cid] if cid in highlight_map else default_color
        ax.hlines(y=y, xmin=x_positions[i], xmax=x_positions[i + 1],
                  colors=color, linewidth=line_width)


def plot_ductal_clone_line_at_x(clone_ids, ax, highlight_map, x=0, line_width=6, default_color="black"):
    """
    Plots a vertical line for the provided clone IDs at horizontal position x.
    The normalized duct length is represented on the y-axis (from 1 at the bottom to 0 at the top).
    The clones are divided equally along this length.
    For each clone, if its ID is in highlight_map then that color is used;
    otherwise, the color given by default_color is used.
    """
    n = len(clone_ids)
    if n == 0:
        return
    cell_height = 1.0 / n
    y_positions = [i * cell_height for i in range(n + 1)]
    for i in range(n - 1):
        cid = str(clone_ids[i])
        color = highlight_map[cid] if cid in highlight_map else default_color
        ax.vlines(x=x, ymin=y_positions[i+1], ymax=y_positions[i],
                  colors=color, linewidth=line_width)


def main():
    do_real_visualization = False
    do_simulated_visualization = True
    do_heatmaps = True
    do_single_duct_plot = True
    do_zoom_plots = True
    do_save_figures = True

    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 14,
        'axes.labelsize': 9,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    random.seed(41)
    output_folder = "output_images_fig_3"
    os.makedirs(output_folder, exist_ok=True)

    # Define common parameters for simulation and visualization.
    # For the pubertal clones, we highlight a few selected ones.
    clone_color_map = {"135": "#ff350e", "40": "#45d555", "122": "#40aef1"}
    subsegments = 100
    vert_gap = 3.5

    if do_real_visualization:
        json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
        duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

        green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
        yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
        red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
        threshold_value = 300

        json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2435322_Phet_24W\clean_and_expanded.json"
        duct_borders_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2435322_Phet_24W\28052024_2435322_L5_ecad_mAX.lif - TileScan 2 Merged_Processed001_outlines.geojson"

        green_image_path = None
        yellow_image_path = None
        red_image_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2435322_Phet_24W\28052024_2435322_L5_ecad_mAX_hyperstackforbranchanalysis-0003.tif"
        threshold_value = 300
        # Load real duct system(s)
        duct_systems = load_duct_systems(json_path)
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

        if os.path.exists(red_image_path):
            red_image = io.imread(red_image_path)
            base_shape = red_image.shape
        else:
            red_image = None
            base_shape = (1024, 1024)

        green_image = io.imread(green_image_path) if green_image_path else None
        yellow_image = io.imread(yellow_image_path) if yellow_image_path else None

        shapes = [(duct_polygon, 1)]
        duct_mask = rasterize(shapes, out_shape=base_shape, fill=0, dtype=np.uint8)

        # Build directed graph for real data.
        G_real = create_directed_duct_graph(duct_system)
        first_bp = list(G_real.nodes())[0]
        while len(list(G_real.predecessors(first_bp))) == 1:
            first_bp = list(G_real.predecessors(first_bp))[0]
        root_node = first_bp
        print(f"Using branch point {root_node} as BFS root")

        # Plot the striped tree for the real network.
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
        if do_save_figures:
            fig_real.savefig(os.path.join(output_folder, "real_network_striped_tree.png"), dpi=600, bbox_inches='tight')
            plt.close(fig_real)

        branch_level_path = os.path.join(output_folder,
                                         "real_network_branch_level_distribution.png") if do_save_figures else None
        plot_branch_level_distribution(
            G_real,
            title="Real Network: Ducts per Branch Level",
            save_path=branch_level_path
        )

    if do_simulated_visualization or do_heatmaps:
        n_clones = 170
        bifurcation_prob = 0.01
        initial_termination_prob = 0.25
        final_termination_prob = 0.55
        max_cells = 3_000_000

        # Simulate Puberty ----------------------------------------------------
        G_puberty, progress_data_puberty = simulate_ductal_tree(
            max_cells=max_cells,
            bifurcation_prob=bifurcation_prob,
            initial_side_count=n_clones / 2,
            initial_center_count=n_clones / 2,
            initial_termination_prob=initial_termination_prob,
            final_termination_prob=final_termination_prob,
        )
        root_node_sim = list(G_puberty.nodes())[0]

        # Choose a set of ducts for downstream analyses (unchanged) -----------
        nodes_puberty = list(G_puberty.nodes())
        if nodes_puberty:
            selected_ducts = [1877]
            while True:
                parents = list(G_puberty.predecessors(selected_ducts[0]))
                if not parents:
                    break
                selected_ducts = parents + selected_ducts
            selected_ducts = selected_ducts[16:]    # from branch level 12
            selected_ducts = selected_ducts[::2]    # every other duct
        else:
            selected_ducts = []

        print("Selected ducts for downstream analyses:", selected_ducts)

        if do_simulated_visualization:
            fig_puberty_uncol, ax_puberty_uncol = plot_hierarchical_graph_subsegments_simulated(
                G_puberty,
                root_node=root_node_sim,
                clone_attr=None,
                annotation_to_color=None,
                subsegments=subsegments,
                use_hierarchy_pos=True,
                vert_gap=4,
                orthogonal_edges=True,
                linewidth=0.9,
                draw_nodes=False
            )
            plt.title("Simulated Ductal Tree after Puberty (Uncoloured)")
            fig_puberty_uncol.set_size_inches(35, 12)
            if do_save_figures:
                fig_puberty_uncol.savefig(
                    os.path.join(output_folder, "puberty_duct_tree_uncoloured.png"),
                    dpi=600, bbox_inches='tight'
                )
                plt.close(fig_puberty_uncol)
            else:
                plt.show()

            # Retain branch-level distribution
            plot_branch_level_distribution(
                G_puberty,
                title="Simulation (Puberty): Ducts per Branch Level",
                save_path=os.path.join(output_folder, "puberty_branch_level_distribution.png")
                if do_save_figures else None
            )

            root_trunc = selected_ducts[0]  # the proximal duct
            print(f"Truncating at root duct: {root_trunc}")
            trunc_nodes = nx.descendants(G_puberty, root_trunc) | {root_trunc}
            G_puberty_trunc = G_puberty.subgraph(trunc_nodes).copy()

            plot_selected_ducts(
                G_puberty_trunc,
                selected_ducts[1:],
                vert_gap=1.8,
                root_node=root_trunc,
                linewidth=14
            )
            plt.title("Simulated Puberty: Selected Ducts (truncated branch)")

            fig_selected = plt.gcf()
            fig_selected.set_size_inches(70, 12)
            if fig_selected.legend():
                fig_selected.legend().remove()
            if do_save_figures:
                save_figure_with_separated_legend(
                    fig_selected,
                    "puberty_selected_ducts_truncated",
                    dpi=600,
                    output_folder=output_folder
                )
                plt.close(fig_selected)

            # Simulate Adulthood (needed for zoom-ins & downstream) ------------
            G_adulthood, progress_data_adult, round_graphs = simulate_adulthood(
                G_puberty.copy(),
                rounds=33,
                output_graphs=True,
                seed=42
            )
            metrics = puberty_vs_adult0(G_puberty, progress_data_adult)
            print(metrics)

            # Still build a colour map for adult clones (needed for zoom-ins).
            adult_clones = {c
                            for u, v in G_adulthood.edges()
                            for c in G_adulthood[u][v].get("adult_clones", [])}
            # random_adult_clones = random.sample(list(adult_clones),
            #                                     min(10_000, len(adult_clones)))
            # palette = sns.color_palette("tab20",
            #                             n_colors=len(random_adult_clones))
            # actually select 2 specific ones
            random_adult_clones = [41626, 139857]
            palette = ["#d097e8", "#F0C807"] # purple and yellow

            clone_color_map_adult = {str(c): palette[i]
                                     for i, c in enumerate(random_adult_clones)}

        else:
            # (This branch unchanged)
            if do_heatmaps:
                G_adulthood, progress_data_adult, round_graphs = simulate_adulthood(
                    G_puberty.copy(),
                    rounds=33,
                    output_graphs=True,
                    seed=42
                )

    if do_simulated_visualization and do_zoom_plots:
        zoom_duct_id = 446

        # Zoom for Puberty (pubertal clones) ----------------------------------
        if zoom_duct_id in G_puberty.nodes():
            zoom_nodes = nx.descendants(G_puberty, zoom_duct_id)
            zoom_nodes.add(zoom_duct_id)
            subG_puberty_zoom = G_puberty.subgraph(zoom_nodes).copy()
            fig_zoom_puberty, _ = plot_hierarchical_graph_subsegments_simulated(
                subG_puberty_zoom,
                root_node=zoom_duct_id,
                clone_attr="duct_clones",
                annotation_to_color=clone_color_map,
                subsegments=subsegments,
                use_hierarchy_pos=True,
                vert_gap=2,
                orthogonal_edges=True,
                linewidth=15,
                draw_nodes=False
            )
            plt.title(f"Zoomed Simulated Ductal Tree after Puberty (Duct {zoom_duct_id})")
            fig_zoom_puberty.set_size_inches(12, 12)
            if do_save_figures:
                save_figure_with_separated_legend(
                    fig_zoom_puberty,
                    f"puberty_duct_tree_zoom_{zoom_duct_id}",
                    dpi=600,
                    output_folder=output_folder
                )
                plt.close(fig_zoom_puberty)
            else:
                plt.show()
        else:
            print(f"Duct {zoom_duct_id} not found in G_puberty.")

        # Zooms for Adulthood (pubertal and adult clones) ---------------------
        if zoom_duct_id in G_adulthood.nodes():
            zoom_nodes_adult = nx.descendants(G_adulthood, zoom_duct_id)
            zoom_nodes_adult.add(zoom_duct_id)
            subG_adulthood_zoom = G_adulthood.subgraph(zoom_nodes_adult).copy()

            # Pubertal clone colours
            fig_zoom_adult_pub, _ = plot_hierarchical_graph_subsegments_simulated(
                subG_adulthood_zoom,
                root_node=zoom_duct_id,
                clone_attr="duct_clones",
                annotation_to_color=clone_color_map,
                subsegments=subsegments,
                use_hierarchy_pos=True,
                vert_gap=2,
                orthogonal_edges=True,
                linewidth=15,
                draw_nodes=False
            )
            plt.title(f"Zoomed Adulthood Tree (Pubertal colours) – Duct {zoom_duct_id}")
            fig_zoom_adult_pub.set_size_inches(12, 12)
            if do_save_figures:
                save_figure_with_separated_legend(
                    fig_zoom_adult_pub,
                    f"adulthood_duct_tree_pubertal_zoom_{zoom_duct_id}",
                    dpi=900,
                    output_folder=output_folder
                )
                plt.close(fig_zoom_adult_pub)
            else:
                plt.show()

            # Adult clone colours
            fig_zoom_adult_adult, _ = plot_hierarchical_graph_subsegments_simulated(
                subG_adulthood_zoom,
                root_node=zoom_duct_id,
                clone_attr="adult_clones",
                annotation_to_color=clone_color_map_adult,
                subsegments=subsegments,
                use_hierarchy_pos=True,
                vert_gap=2,
                orthogonal_edges=True,
                linewidth=15,
                draw_nodes=False
            )
            plt.title(f"Zoomed Adulthood Tree (Adult colours) – Duct {zoom_duct_id}")
            fig_zoom_adult_adult.set_size_inches(12, 12)
            if do_save_figures:
                save_figure_with_separated_legend(
                    fig_zoom_adult_adult,
                    f"adulthood_duct_tree_adult_zoom_{zoom_duct_id}",
                    dpi=900,
                    output_folder=output_folder
                )
                plt.close(fig_zoom_adult_adult)
            else:
                plt.show()
        else:
            print(f"Duct {zoom_duct_id} not found in G_adulthood.")

    if do_heatmaps:
        # pop the first one from selected ducts
        selected_ducts = selected_ducts[1:]
        df_puberty = gather_clone_fractions_for_selected_ducts(
            G_puberty, selected_ducts, puberty_clones=True
        )
        df_adulthood_pubertal = gather_clone_fractions_for_selected_ducts(
            G_adulthood, selected_ducts, puberty_clones=True
        )
        df_adulthood_adult = gather_clone_fractions_for_selected_ducts(
            G_adulthood, selected_ducts, puberty_clones=False, restrict_to_selected=True
        )

        vaf_threshold = 0
        vaf_columns = selected_ducts
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

        df_puberty['Pattern'] = df_puberty.apply(assign_pattern, axis=1)
        df_puberty['Cluster'] = df_puberty['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))
        df_puberty.sort_values('Cluster', inplace=True)
        nonzero_pattern = '0' * len(vaf_columns)
        df_puberty = df_puberty[df_puberty['Pattern'] != nonzero_pattern]
        df_adulthood_pubertal = df_adulthood_pubertal.reindex(df_puberty.index, fill_value=0.0)
        df_adulthood_adult['Pattern'] = df_adulthood_adult.apply(assign_pattern, axis=1)
        df_adulthood_adult['Cluster'] = df_adulthood_adult['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))
        df_adulthood_adult.sort_values('Cluster', inplace=True)
        df_adulthood_adult = df_adulthood_adult[df_adulthood_adult['Pattern'] != nonzero_pattern]

        plt.figure(figsize=(5, 5))
        sns.heatmap(df_puberty[selected_ducts].T, vmin=0.0, vmax=0.2)
        plt.ylabel("Duct ID")
        plt.xlabel("Clone IDs")
        plt.xticks([])
        plt.title("Post-Puberty")
        if do_save_figures:
            plt.savefig(os.path.join(output_folder, "heatmap_puberty_clone_fractions.png"), dpi=900,
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(5, 5))
        sns.heatmap(df_adulthood_pubertal[selected_ducts].T, vmin=0.0, vmax=0.2)
        plt.ylabel("Duct ID")
        plt.xlabel("Clone IDs (Pubertal)")
        plt.xticks([])
        plt.title("Post-Adulthood")
        if do_save_figures:
            plt.savefig(os.path.join(output_folder, "heatmap_adulthood_pubertal_clone_fractions.png"), dpi=900,
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(20, 5))
        sns.heatmap(df_adulthood_adult[selected_ducts].T, vmin=0.0, vmax=0.2)
        plt.ylabel("Duct ID")
        plt.xlabel("Clone IDs (Adult)")
        # plt.xticks([])
        plt.title("Simulated Sequencing Post-Adulthood (Adult Clones)")
        if do_save_figures:
            plt.savefig(os.path.join(output_folder, "heatmap_adulthood_adult_clone_fractions.png"), dpi=900,
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    if do_single_duct_plot:
        duct_id = 1877  # duct to visualize
        n_iterations = 33  # number of adulthood iterations

        # Add the final state of puberty (pubertal clones are stored as "duct_clones" in G_puberty)
        puberty_clones_final = gather_clones_for_duct(G_puberty, duct_id, puberty_clones=True)
        pubertal_clones_over_iterations = [puberty_clones_final]
        print(f"Puberty state: {len(puberty_clones_final)} pubertal clones.")

        # Then add the snapshots from the adulthood simulation (round_graphs[0] is the state after 0 rounds)
        for i in range(n_iterations + 1):
            snapshot = round_graphs[i]
            clones_pubertal = gather_clones_for_duct(snapshot, duct_id, puberty_clones=True)
            pubertal_clones_over_iterations.append(clones_pubertal)
            print(f"Iteration {i}: {len(clones_pubertal)} pubertal clones.")
        total_pubertal = len(pubertal_clones_over_iterations)
        fig_single_pubertal, ax_single_pubertal = plt.subplots(figsize=(11, 8))

        # Pubertal state (thick line) at x = –2
        plot_ductal_clone_line_at_x(
            pubertal_clones_over_iterations[0],
            ax_single_pubertal,
            clone_color_map,
            x=-2,
            line_width=17,
            default_color="black",
        )

        # Adulthood iterations (also thick lines) at x = 0, 1, …, n_iterations
        for i in range(1, total_pubertal):
            x_val = i - 1
            plot_ductal_clone_line_at_x(
                pubertal_clones_over_iterations[i],
                ax_single_pubertal,
                clone_color_map,
                x=x_val,
                line_width=17,
                default_color="black",
            )

        wanted_ticks = list(range(5, n_iterations + 1, 5))  # 5, 10, 15, …
        if n_iterations not in wanted_ticks:  # ensure the final (e.g. 33) is shown
            wanted_ticks.append(n_iterations)

        xticks = [-2] + wanted_ticks
        xtick_labels = ["pubertal"] + [str(i) for i in wanted_ticks]

        ax_single_pubertal.set_xticks(xticks)
        ax_single_pubertal.set_xticklabels(xtick_labels, fontsize=28)  # bump font size

        ax_single_pubertal.set_xlabel("Iteration")
        ax_single_pubertal.set_ylabel("Normalized Duct Length")
        ax_single_pubertal.set_ylim(1, 0)
        ax_single_pubertal.set_title(
            "Evolution of Pubertal Clones Over Adulthood Iterations (Single Duct)"
        )
        plt.tight_layout()

        if do_save_figures:
            fig_single_pubertal.savefig(os.path.join(output_folder, "single_duct_pubertal.png"), dpi=600,
                                         bbox_inches="tight")
            plt.close(fig_single_pubertal)

        adult_clones_over_iterations = []
        for i in range(n_iterations + 1):
            snapshot = round_graphs[i]
            clones_adult = gather_clones_for_duct(snapshot, duct_id, puberty_clones=False)
            adult_clones_over_iterations.append(clones_adult)
            print(f"Iteration {i}: {len(clones_adult)} adult clones.")

        total_adult = len(adult_clones_over_iterations)
        # Create a dedicated color map covering all adult clone IDs.
        all_adult_ids = set()
        for clones in adult_clones_over_iterations:
            all_adult_ids.update(map(str, clones))
        all_adult_ids = list(all_adult_ids)
        palette = sns.color_palette("tab20", n_colors=len(all_adult_ids))
        # shuffle the palette for better visual distinction
        random.shuffle(all_adult_ids)
        adult_color_map = {cid: palette[i] for i, cid in enumerate(all_adult_ids)}

        fig_single_adult, ax_single_adult = plt.subplots(figsize=(11, 8))
        for i, clones_iter in enumerate(adult_clones_over_iterations):
            plot_ductal_clone_line_at_x(clones_iter, ax_single_adult, adult_color_map,
                                        x=i, line_width=17, default_color="black")
        ax_single_adult.set_xticks(range(total_adult))
        ax_single_adult.set_xticklabels([str(i) for i in range(total_adult)])
        ax_single_adult.set_xlabel("Iteration")
        ax_single_adult.set_ylabel("Normalized Duct Length")
        ax_single_adult.set_ylim(1, 0)
        ax_single_adult.set_title("Evolution of Adult Clones Over Adulthood Iterations (Single Duct)")
        plt.tight_layout()
        if do_save_figures:
            fig_single_adult.savefig(os.path.join(output_folder, "single_duct_adult.png"), dpi=600,
                                     bbox_inches="tight")
            plt.close(fig_single_adult)

    # Finally, if saving images is on, close all figures; otherwise, display them.
    if do_save_figures:
        plt.close('all')
    else:
        plt.show()


if __name__ == "__main__":
    main()
