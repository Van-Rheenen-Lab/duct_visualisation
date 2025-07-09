import json
import networkx as nx
import warnings
from shapely.geometry import LineString
import numpy as np

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

def create_duct_graph(duct_system):
    """
    Create a networkx graph from the given duct system.
    Each edge now stores additional metadata including internal_points, z-values,
    annotations, and properties (which can include an annotation label).
    """
    G = nx.Graph()
    bps = duct_system["branch_points"]
    segs = duct_system["segments"]

    # Add nodes with their attributes.
    for bp_name, bp_data in bps.items():
        G.add_node(bp_name, **bp_data)
        # Also store the position as a tuple.
        G.nodes[bp_name]["pos"] = (bp_data["x"], bp_data["y"], bp_data["z"])

    # Add edges along with full metadata.
    for seg_name, seg_data in segs.items():
        start_bp = seg_data["start_bp"]
        end_bp = seg_data["end_bp"]
        if start_bp not in G or end_bp not in G:
            warnings.warn(f"Segment '{seg_name}' references undefined branch points.")
            continue

        G.add_edge(
            start_bp,
            end_bp,
            segment_name=seg_name,
            internal_points=seg_data.get('internal_points', []),
            start_z=seg_data.get('start_z'),
            end_z=seg_data.get('end_z'),
            annotations=seg_data.get('annotations', []),
            properties=seg_data.get('properties', {})  # This can include your "Annotation" key.
        )

    return G


def create_directed_duct_graph(duct_system: dict) -> nx.DiGraph:
    """
    Creates a directed graph with nodes for all branch_points and
    directed edges for each segment. All metadata from duct_system is stored
    in the node and edge attributes.
    """
    G_dir = nx.DiGraph()
    branch_points = duct_system.get("branch_points", {})
    segments = duct_system.get("segments", {})

    # Add branch points as nodes.
    for bp_name, bp_data in branch_points.items():
        G_dir.add_node(bp_name, **bp_data)

    # Add edges for segments.
    for seg_name, seg_data in segments.items():
        start_bp = seg_data["start_bp"]
        end_bp = seg_data["end_bp"]

        if start_bp not in branch_points or end_bp not in branch_points:
            warnings.warn(f"Segment '{seg_name}' references undefined branch points.")
            continue

        # Store all segment properties on the edge.
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


def get_line_for_edge(G: nx.DiGraph, u: str, v: str) -> LineString:
    """
    Build a LineString for the edge between nodes u and v using node and edge attributes.
    """
    # Get coordinates from the nodes.
    start_data = G.nodes[u]
    end_data = G.nodes[v]
    pts = [(start_data['x'], start_data['y'])]

    # Optionally include internal points stored on the edge.
    edge_data = G.get_edge_data(u, v)
    if edge_data and edge_data.get("internal_points"):
        pts.extend([(p['x'], p['y']) for p in edge_data.get("internal_points", [])])
    pts.append((end_data['x'], end_data['y']))

    return LineString(pts)

def find_root(G: nx.DiGraph) -> str:
    """
    Determine a good root for a traversal by starting at the first branch point and
    moving upstream until a node with either no predecessor or multiple predecessors is found.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph representing the duct system.

    Returns
    -------
    str
        The root node.
    """
    if not G.nodes:
        raise ValueError("Graph is empty; cannot determine a root.")

    # Start with the first branch point.
    current_node = list(G.nodes())[0]

    # Move upstream while there predecessors.
    while len(list(G.predecessors(current_node))) >= 1:
        current_node = list(G.predecessors(current_node))[0]

    print(f"Using branch point '{current_node}' as root")
    return current_node


def select_biggest_duct_system(duct_systems: dict) -> tuple:
    """
    Select the duct system with the most branch points from a dictionary of duct systems.
    """
    selected_index = None
    max_bps = 0

    for key, system in duct_systems.items():
        num_bps = len(system.get("branch_points", {}))
        if num_bps > max_bps:
            max_bps = num_bps
            selected_index = key

    if selected_index is None:
        raise ValueError("No duct systems with branch points were found.")

    print(f"Using network with key '{selected_index}' as the graph")
    return duct_systems[selected_index]

def load_duct_mask(duct_borders_path, out_shape, buffer=0, all_touched=False):
    """
    Loads duct border geometries from a GeoJSON file, fixes invalid geometries,
    unions them into a single polygon, and rasterizes the result to create a mask.

    Parameters
    ----------
    duct_borders_path : str
        Path to the GeoJSON file containing duct borders.
    out_shape : tuple
        The shape (height, width) for the output raster mask.
    buffer : float, optional
        Buffer distance to apply to the unioned polygon before rasterizing.
    all_touched : bool, optional
        Passed to rasterize().

    Returns
    -------
    duct_polygon : shapely.geometry.Polygon
        The unioned (and optionally buffered) duct polygon.
    duct_mask : np.ndarray
        A binary mask of shape `out_shape` with the duct area.
    """
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    if duct_borders_path is None:
        h, w = out_shape
        full_poly = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
        full_mask = np.ones(out_shape, dtype=np.uint8)
        return full_poly, full_mask

    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)

    valid_geoms = []
    for feat in duct_borders['features']:
        geom = shape(feat['geometry'])
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_valid:
            valid_geoms.append(geom)
        else:
            print("Skipping geometry that could not be fixed:", feat['geometry'])

    duct_polygon = unary_union(valid_geoms)
    if buffer:
        duct_polygon = duct_polygon.buffer(buffer)

    duct_mask = rasterize(
        [(duct_polygon, 1)],
        out_shape=out_shape,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched
    )
    return duct_polygon, duct_mask

