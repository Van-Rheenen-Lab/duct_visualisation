import networkx as nx
from analysis.utils.loading_saving import find_root
from typing import Dict, Any, Optional, Set

def simplify_graph(G_dir: nx.DiGraph, main_branch_node: Optional[str] = None) -> nx.DiGraph:
    """
    Very soft simplification:
    - Only collapse a node B if it has exactly one predecessor and one successor (A->B->C),
      is not the root, and has degree 2 in the undirected graph.
    - Collapse A->B->C into A->C, merging annotations/properties.
    - Do NOT collapse direct edges between branch points or longer chains.
    """
    if main_branch_node is None:
        main_branch_node = find_root(G_dir)
    if main_branch_node not in G_dir:
        raise ValueError(f"Main branch node '{main_branch_node}' not found in the duct system.")

    G_clean = G_dir.copy()
    G_undir = G_clean.to_undirected()
    nodes_removed = True
    while nodes_removed:
        nodes_removed = False
        candidates = [
            node for node in list(G_clean.nodes())
            if node != main_branch_node
            and (G_clean.in_degree(node) + G_clean.out_degree(node) == 2)
            and G_undir.degree(node) == 2
        ]
        for node in candidates:
            pred = next(G_clean.predecessors(node), None)
            succ = next(G_clean.successors(node), None)
            # If both are None, skip
            if pred is None or succ is None:
                # Try the other direction (for undirected robustness)
                neighbors = list(G_clean.neighbors(node))
                if len(neighbors) == 2:
                    pred, succ = neighbors
                else:
                    continue
            if pred == succ:
                continue
            # Merge edge data
            edge1 = G_clean.get_edge_data(pred, node, default={}) or G_clean.get_edge_data(node, pred, default={})
            edge2 = G_clean.get_edge_data(node, succ, default={}) or G_clean.get_edge_data(succ, node, default={})
            ann1 = edge1.get('annotations', [])
            ann2 = edge2.get('annotations', [])
            merged_annotations = list({*ann1, *ann2})
            prop1 = edge1.get('properties', {})
            prop2 = edge2.get('properties', {})
            prop_ann1 = prop1.get('Annotation', [])
            prop_ann2 = prop2.get('Annotation', [])
            if isinstance(prop_ann1, str):
                prop_ann1 = [prop_ann1]
            if isinstance(prop_ann2, str):
                prop_ann2 = [prop_ann2]
            merged_prop_ann = list({*prop_ann1, *prop_ann2})
            if (ann1 and ann2) or (prop_ann1 and prop_ann2):
                print(f"[simplify_graph] WARNING: Both edges {pred}->{node} and {node}->{succ} have annotations. Merging all.")
            merged_properties = {**prop1, **prop2}
            if merged_prop_ann:
                merged_properties['Annotation'] = merged_prop_ann
            new_edge_data = {
                "segment_name": f"{pred}to{succ}",
                "annotations": merged_annotations,
                "properties": merged_properties
            }
            if not G_clean.has_edge(pred, succ):
                G_clean.add_edge(pred, succ, **new_edge_data)
            G_clean.remove_node(node)
            nodes_removed = True
        G_undir = G_clean.to_undirected()
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

def orient_graph_from_root(G_dir: nx.DiGraph, root: str) -> nx.DiGraph:
    """
    Return a DiGraph where EVERY edge is oriented away from `root`.
    Works even if some original edges point toward the root.
    """
    if root not in G_dir:
        raise ValueError(f"Root node '{root}' not found in graph.")

    G_oriented = nx.DiGraph()
    visited    = {root}
    queue      = [root]

    G_undir = G_dir.to_undirected()

    # ---------- breadth-first ‘paint away from root’ ----------
    while queue:
        parent = queue.pop(0)

        # ensure the parent itself is present in the result
        if parent not in G_oriented:
            G_oriented.add_node(parent)

        for child in G_undir.neighbors(parent):
            if child in visited:
                continue

            # copy any edge data we can find (either direction)
            data = (G_dir.get_edge_data(parent, child, default={})
                    or G_dir.get_edge_data(child, parent, default={})
                    or {})

            G_oriented.add_edge(parent, child, **data)

            visited.add(child)
            queue.append(child)

    for n, d in G_dir.nodes(data=True):
        if n not in G_oriented:
            G_oriented.add_node(n)
        G_oriented.nodes[n].update(d)

    return G_oriented
