import networkx as nx
from analysis.utils.loading_saving import find_root
from typing import Dict, Any, Optional, Set

def simplify_graph(G_dir: nx.DiGraph, main_branch_node: Optional[str] = None) -> nx.DiGraph:
    """
    Simplify a directed duct graph by iteratively removing intermediate nodes
    (nodes with exactly one incoming and one outgoing edge) except the main branch node.
    When a node is removed, a new edge is added that merges the properties of the two edges.

    Parameters
    ----------
    G_dir : nx.DiGraph
        The original directed duct graph.
    main_branch_node : Optional[str]
        If not provided, determine a root automatically.

    Returns
    -------
    nx.DiGraph
        A new directed graph with intermediate nodes removed.
    """
    # Suppose find_root is a function that determines a good root (BFS) from the graph.
    if main_branch_node is None:
        main_branch_node = find_root(G_dir)
    if main_branch_node not in G_dir:
        raise ValueError(f"Main branch node '{main_branch_node}' not found in the duct system.")

    G_clean = G_dir.copy()
    nodes_removed = True
    while nodes_removed:
        nodes_removed = False
        # Candidate nodes have one predecessor and one successor.
        candidates = [node for node in list(G_clean.nodes())
                      if node != main_branch_node and G_clean.in_degree(node) == 1 and G_clean.out_degree(node) == 1]
        for node in candidates:
            pred = next(G_clean.predecessors(node))
            succ = next(G_clean.successors(node))
            if pred == succ:
                continue

            # Create a new segment name and merge properties as needed.
            new_segment_name = f"{pred}to{succ}"
            new_edge_data = {"segment_name": new_segment_name}
            # (Optionally, merge other attributes from edges pred->node and node->succ)

            if not G_clean.has_edge(pred, succ):
                G_clean.add_edge(pred, succ, **new_edge_data)

            # Remove the intermediate node (which also removes its incident edges).
            G_clean.remove_node(node)
            nodes_removed = True

    return G_clean


def connect_component_to_main(G: nx.Graph,
                              main_component: Set[str],
                              comp_component: Set[str],
                              coord_tol: float = 10) -> bool:
    """
    Attempt to unify a disconnected component into the main network by merging branch points
    that have essentially the same coordinates (within coord_tol). Branch point coordinates
    are stored as node attributes "x", "y", and optionally "z" (defaulting to 0 if absent).

    Parameters
    ----------
    G : nx.Graph
        The undirected graph of the duct system.
    main_component : Set[str]
        The set of nodes in the main component.
    comp_component : Set[str]
        The set of nodes in the disconnected component.
    coord_tol : float, optional
        Tolerance for coordinate matching, by default 10.

    Returns
    -------
    bool
        True if a merge was performed; False otherwise.
    """
    # Build coordinates for main component nodes.
    main_coords = {
        node: (
            G.nodes[node].get("x"),
            G.nodes[node].get("y"),
            G.nodes[node].get("z", 0)
        )
        for node in main_component
        if "x" in G.nodes[node] and "y" in G.nodes[node]
    }

    def same_coord(p1, p2):
        return (abs(p1[0] - p2[0]) < coord_tol and
                abs(p1[1] - p2[1]) < coord_tol and
                abs(p1[2] - p2[2]) < coord_tol)

    match_node_main = None
    match_node_comp = None

    # Look for a node in comp_component with matching coordinates.
    for cnode in comp_component:
        if "x" not in G.nodes[cnode] or "y" not in G.nodes[cnode]:
            continue
        comp_coord = (
            G.nodes[cnode].get("x"),
            G.nodes[cnode].get("y"),
            G.nodes[cnode].get("z", 0)
        )
        for mnode, mcoord in main_coords.items():
            if same_coord(comp_coord, mcoord):
                match_node_main = mnode
                match_node_comp = cnode
                break
        if match_node_main is not None:
            break

    if match_node_main is None:
        # No matching branch point found.
        return False

    # Rewire edges: For every neighbor of match_node_comp,
    # add an edge from match_node_main if it doesn't exist.
    neighbors = list(G.neighbors(match_node_comp))
    for nbr in neighbors:
        if nbr == match_node_main:
            continue  # Skip self-loop.
        edge_data = G.get_edge_data(match_node_comp, nbr)
        if not G.has_edge(match_node_main, nbr):
            G.add_edge(match_node_main, nbr, **edge_data)
        else:
            # Optionally, merge edge attributes if needed.
            pass
        G.remove_edge(match_node_comp, nbr)

    # Remove the merged node.
    G.remove_node(match_node_comp)

    return True
