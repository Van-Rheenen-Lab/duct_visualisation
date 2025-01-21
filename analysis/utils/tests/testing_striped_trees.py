import json
import matplotlib.pyplot as plt
from analysis.utils.loading_saving import create_directed_duct_graph
from shapely.geometry import shape
from shapely.validation import make_valid
from analysis.utils.loading_saving import load_duct_systems


def plotting_outline_and_annotation(json_path, duct_borders_path):
    duct_systems = load_duct_systems(json_path)
    duct_system = duct_systems[1]

    G = create_directed_duct_graph(duct_system)

    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)

    valid_geoms = []
    for feature in duct_borders['features']:
        geom = shape(feature['geometry'])

        # Attempt to fix geometry if invalid
        if not geom.is_valid:
            geom = make_valid(geom)

        if geom.is_valid:
            valid_geoms.append(geom)
        else:
            print("Skipping geometry that could not be fixed:", feature['geometry'])

    plt.figure(figsize=(8, 6))
    for poly in valid_geoms:
        # If you had minor geometry issues, you could do poly.buffer(0), but if it's valid, skip
        poly_clean = poly.buffer(0)

        # Exterior boundary
        x, y = poly_clean.exterior.xy
        plt.plot(x, y, 'b-', label="Duct Outline" if 'label' not in plt.gca().lines else "")

        # Interior holes in the polygon
        for hole in poly_clean.interiors:
            hx, hy = hole.xy
            plt.plot(hx, hy, 'b-')

    # Plot edges
    for u, v in G.edges():
        x_vals = [G.nodes[u]["x"]]
        y_vals = [G.nodes[u]["y"]]

        # internal points
        if "internal_points" in G[u][v]:
            for internal_point in G[u][v]["internal_points"]:
                x_vals.append([internal_point][0]["x"])
                y_vals.append([internal_point][0]["y"])

        x_vals.append(G.nodes[v]["x"])
        y_vals.append(G.nodes[v]["y"])

        plt.plot(x_vals, y_vals, 'r', linewidth=1, label="Duct Edge" if 'label' not in plt.gca().lines else "")

    plt.title("Duct Outlines and Network Overlay")
    plt.axis('equal')


if __name__ == "__main__":
    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    plotting_outline_and_annotation(json_path, duct_borders_path)
    plt.show()
