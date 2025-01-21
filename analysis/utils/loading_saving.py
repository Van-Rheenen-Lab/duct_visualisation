import json
import networkx as nx
import warnings
import matplotlib.pyplot as plt

def load_duct_systems(json_path):
    """
    Load duct systems from a JSON file into a structured dictionary.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        {system_index: {"branch_points": dict, "segments": dict}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "duct_systems" not in data:
        raise ValueError("JSON file does not contain 'duct_systems'.")

    duct_systems = {}
    for i, system in enumerate(data["duct_systems"]):
        if "branch_points" not in system or "segments" not in system:
            raise ValueError(f"Duct system at index {i} is missing 'branch_points' or 'segments'.")
        duct_systems[i] = {
            "branch_points": system["branch_points"],
            "segments": system["segments"]
        }

    return duct_systems

#
# def create_duct_graph(duct_system):
#     """
#     Create a networkx graph from the given duct system.
#
#     Parameters
#     ----------
#     duct_system : dict
#         Dictionary containing "branch_points" and "segments".
#
#     Returns
#     -------
#     nx.Graph
#         A graph where nodes are branch points and edges are segments.
#     """
#     G = nx.Graph()
#     bps = duct_system["branch_points"]
#     segs = duct_system["segments"]
#
#     # Add nodes
#     for bp_name in bps:
#         G.add_node(bp_name, **bps[bp_name])
#         # add their position as an attribute
#         G.nodes[bp_name]["pos"] = (bps[bp_name]["x"], bps[bp_name]["y"], bps[bp_name]["z"])
#
#     # Add edges
#     for seg_name, seg_data in segs.items():
#         start_bp = seg_data["start_bp"]
#         end_bp = seg_data["end_bp"]
#         if start_bp not in G or end_bp not in G:
#             warnings.warn(f"Segment '{seg_name}' references undefined branch points.")
#             continue
#         G.add_edge(start_bp, end_bp, segment_name=seg_name)
#
#     return G

def create_directed_duct_graph(duct_system):
    """
    Creates a directed graph with nodes for all branch_points,
    and directed edges from start_bp -> end_bp for each segment.

    If a segment references an undefined branch point, logs a warning
    and skips that segment.
    """
    G_dir = nx.DiGraph()
    branch_points = duct_system.get("branch_points", {})
    segments = duct_system.get("segments", {})

    # Add all branch points as nodes with attributes
    for bp_name, bp_data in branch_points.items():
        G_dir.add_node(bp_name, **bp_data)

    # Add directed edges for valid segments
    for seg_name, seg_data in segments.items():
        start_bp = seg_data["start_bp"]
        end_bp = seg_data["end_bp"]

        if start_bp not in branch_points or end_bp not in branch_points:
            warnings.warn(f"Segment '{seg_name}' references undefined branch points.")
            continue

        G_dir.add_edge(
            start_bp,
            end_bp,
            segment_name=seg_name,
            internal_points=seg_data.get('internal_points', []),
            start_z=seg_data.get('start_z'),
            end_z=seg_data.get('end_z'),
            annotations=seg_data.get('annotations', []),
            properties=seg_data.get('properties', {})
        )

    return G_dir



def save_annotations(duct_systems, filename):
    """
    Save the fixed (modified) duct systems back into a JSON file.
    """
    data = {'duct_systems': []}
    for system_data in duct_systems.values():
        data['duct_systems'].append({
            'branch_points': {
                name: {
                    'x': bp['x'],
                    'y': bp['y'],
                    'z': bp['z']
                } for name, bp in system_data['branch_points'].items()
            },
            'segments': {
                name: {
                    'start_bp': seg_data['start_bp'],
                    'end_bp': seg_data['end_bp'],
                    'internal_points': seg_data['internal_points'],
                    'start_z': seg_data['start_z'],
                    'end_z': seg_data['end_z'],
                    'annotations': seg_data.get('annotations', []),
                    'properties': seg_data.get('properties', {})
                } for name, seg_data in system_data['segments'].items()
            }
        })

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    # Path to your JSON file
    from fixing_annotations import simplify_duct_system
    from plotting_trees import plot_hierarchical_graph
    json_path = r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\T18_05203.json"

    # Load duct systems
    duct_systems = load_duct_systems(json_path)

    plot_hierarchical_graph(create_duct_graph(duct_systems[4]), use_hierarchy_pos=True, orthogonal_edges=True)

    duct_graphs = []
    for idx, system_data in duct_systems.items():
        # Clean initial data
        if len(system_data["segments"]) > 0:

            # main branch node is the first branch point
            main_branch_node = list(system_data["branch_points"].keys())[0]

            # Simplify the duct system
            try:
                system_data = simplify_duct_system(system_data, main_branch_node)
            except ValueError as e:
                warnings.warn(f"System {idx}: {e}")
                continue

            # Create graph from the simplified duct system
            G = create_duct_graph(system_data)


            duct_graphs.append(G)

    plot_hierarchical_graph(duct_graphs[0], use_hierarchy_pos=True, orthogonal_edges=True, vert_gap=1, vert_length=1)
    plt.show()