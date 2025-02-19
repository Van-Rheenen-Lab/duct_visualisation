import networkx as nx
from analysis.utils.loading_saving import create_directed_duct_graph
import warnings

def simplify_duct_system(duct_system, main_branch_node):
    """
    Simplify the duct system by removing intermediate nodes (nodes with only one child)
    and ensuring all nodes are connected to the main branch.

    Parameters
    ----------
    duct_system : dict
        Dictionary containing "branch_points" and "segments".
    main_branch_node : str
        The identifier of the main branch node (e.g., "bp0").

    Returns
    -------
    dict
        The simplified duct system.
    """
    # Create a graph from the duct system
    G = create_directed_duct_graph(duct_system)

    # Ensure the main_branch_node exists
    if main_branch_node not in G:
        raise ValueError(f"Main branch node '{main_branch_node}' not found in the duct system.")

    # Simplify the graph by removing intermediate nodes
    G_simplified = nx.Graph(G)  # Make a copy to avoid modifying the original graph

    # Identify nodes to remove: degree 2 and not the main branch node
    nodes_to_remove = [n for n, d in G_simplified.degree() if d == 2 and n != main_branch_node]

    for node in nodes_to_remove:
        neighbors = list(G_simplified.neighbors(node))
        if len(neighbors) != 2:
            continue  # Safety check

        u, v = neighbors

        # Retrieve segment names for the connecting segments
        seg_u = None
        seg_v = None
        for seg_name, seg_data in duct_system["segments"].items():
            if (seg_data["start_bp"] == u and seg_data["end_bp"] == node) or \
               (seg_data["start_bp"] == node and seg_data["end_bp"] == u):
                seg_u = seg_name
            if (seg_data["start_bp"] == v and seg_data["end_bp"] == node) or \
               (seg_data["start_bp"] == node and seg_data["end_bp"] == v):
                seg_v = seg_name

        # Remove the intermediate node
        G_simplified.remove_node(node)

        # Add a new segment connecting u and v
        new_seg_name = f"{seg_u}_{seg_v}_simplified"
        G_simplified.add_edge(u, v, segment_name=new_seg_name)

        # Remove old segments from duct_system
        if seg_u:
            del duct_system["segments"][seg_u]
        if seg_v:
            del duct_system["segments"][seg_v]

        # Add the new simplified segment
        duct_system["segments"][new_seg_name] = {
            "start_bp": u,
            "end_bp": v
            # Add other necessary attributes here if needed
        }

    # Update branch points by removing the simplified intermediate nodes
    for node in nodes_to_remove:
        if node in duct_system["branch_points"]:
            del duct_system["branch_points"][node]

    # Check connectivity to the main branch node
    connected_nodes = nx.node_connected_component(G_simplified, main_branch_node)
    all_nodes = set(G_simplified.nodes())
    if connected_nodes != all_nodes:
        disconnected = all_nodes - connected_nodes
        warnings.warn(
            f"The following nodes are not connected to the main branch '{main_branch_node}': {disconnected}"
        )

    return duct_system

def connect_component_to_main(G, main_component, comp_component, system_data, coord_tol=10):
    """
    Attempt to unify a disconnected component into the main network by merging branch points
    that have essentially the same coordinates (within coord_tol).
    """
    # Extract coordinates for main_component
    main_coords = {
        node: (
            system_data["branch_points"][node]["x"],
            system_data["branch_points"][node]["y"],
            system_data["branch_points"][node]["z"]
        )
        for node in main_component
    }

    def same_coord(p1, p2):
        return (abs(p1[0] - p2[0]) < coord_tol and
                abs(p1[1] - p2[1]) < coord_tol and
                abs(p1[2] - p2[2]) < coord_tol)

    match_node_main = None
    match_node_comp = None

    # Try to find a matching node by coordinates
    for cnode in comp_component:
        cx = system_data["branch_points"][cnode]["x"]
        cy = system_data["branch_points"][cnode]["y"]
        cz = system_data["branch_points"][cnode]["z"]
        comp_coord = (cx, cy, cz)
        for mnode, mcoord in main_coords.items():
            if same_coord(comp_coord, mcoord):
                match_node_main = mnode
                match_node_comp = cnode
                break
        if match_node_main is not None:
            break

    if match_node_main is None:
        # No match found
        return False

    # Unify nodes
    # 1. Update segments in system_data
    for seg_name, seg_data in system_data['segments'].items():
        if seg_data['start_bp'] == match_node_comp:
            seg_data['start_bp'] = match_node_main
        if seg_data['end_bp'] == match_node_comp:
            seg_data['end_bp'] = match_node_main

    # 2. Move edges from match_node_comp to match_node_main in the graph
    neighbors = list(G.neighbors(match_node_comp))
    for nbr in neighbors:
        edge_data = G.get_edge_data(match_node_comp, nbr)
        if not G.has_edge(match_node_main, nbr):
            G.add_edge(match_node_main, nbr, **edge_data)
        G.remove_edge(match_node_comp, nbr)

    # Remove the old node
    G.remove_node(match_node_comp)
    del system_data["branch_points"][match_node_comp]

    return True
