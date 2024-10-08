import json
import networkx as nx
import matplotlib.pyplot as plt
import community  # Install with 'pip install python-louvain'

# Define functions to extract numbers from bp names
def extract_bp_number(bp_name):
    if bp_name.startswith('bp'):
        number_part = bp_name[2:]
        if number_part.isdigit():
            return int(number_part)
    return None  # Return None if not a standard 'bp' name

# Load the JSON data
with open(r'C:/Users/j.doornbos/Downloads/annotations_1_corruption_amputated.json', 'r') as file:
    data = json.load(file)

# We'll process each duct system in the data
for system_index, duct_system in enumerate(data["duct_systems"]):
    branch_points = duct_system['branch_points']
    segments = duct_system['segments']

    # Collect existing 'bp' numbers to avoid duplication
    existing_bp_numbers = set()
    for bp_name in branch_points.keys():
        bp_number = extract_bp_number(bp_name)
        if bp_number is not None:
            existing_bp_numbers.add(bp_number)

    # Find the maximum existing 'bp' number
    max_bp_number = max(existing_bp_numbers) if existing_bp_numbers else 0
    next_bp_number = max_bp_number + 1

    # Mapping from original names to new normalized names
    name_mapping = {}
    reverse_name_mapping = {}

    # Assign new unique 'bp' numbers to inconsistent branch point names
    for original_bp_name in list(branch_points.keys()):
        if original_bp_name.startswith('bp'):
            # Name is already consistent
            new_bp_name = original_bp_name
            name_mapping[original_bp_name] = new_bp_name
            reverse_name_mapping[new_bp_name] = original_bp_name
        else:
            # Name is inconsistent, assign a new unique 'bp' number
            while next_bp_number in existing_bp_numbers:
                next_bp_number += 1  # Ensure uniqueness
            new_bp_name = f'bp{next_bp_number}'
            next_bp_number += 1
            existing_bp_numbers.add(int(new_bp_name[2:]))
            # Update the branch point name in branch_points
            branch_points[new_bp_name] = branch_points.pop(original_bp_name)
            branch_points[new_bp_name]['name'] = new_bp_name  # Update name inside bp data
            name_mapping[original_bp_name] = new_bp_name
            reverse_name_mapping[new_bp_name] = original_bp_name
            print(f"Renamed branch point '{original_bp_name}' to '{new_bp_name}' to avoid overlap.")

    # Update segments with new branch point names
    new_segments = {}
    for segment_name, segment_data in segments.items():
        start_bp = segment_data['start_bp']
        end_bp = segment_data['end_bp']
        new_start_bp = name_mapping.get(start_bp, start_bp)
        new_end_bp = name_mapping.get(end_bp, end_bp)
        segment_data['start_bp'] = new_start_bp
        segment_data['end_bp'] = new_end_bp

        # Update segment name
        new_segment_name = f"{new_start_bp}to{new_end_bp}"
        new_segments[new_segment_name] = segment_data

    duct_system['segments'] = new_segments

    # Remove unconnected branch points
    connected_bps = set()
    for segment_data in duct_system['segments'].values():
        connected_bps.add(segment_data['start_bp'])
        connected_bps.add(segment_data['end_bp'])

    unconnected_bps = set(branch_points.keys()) - connected_bps
    for unconnected_bp in unconnected_bps:
        del branch_points[unconnected_bp]
        print(f"Removed unconnected branch point '{unconnected_bp}'.")

    # Create the graph with normalized names
    G = nx.Graph()
    G.add_nodes_from(branch_points.keys())
    for segment_data in duct_system['segments'].values():
        G.add_edge(segment_data['start_bp'], segment_data['end_bp'])

    # Ensure the graph is a tree
    if nx.is_tree(G):
        T = G
    else:
        T = nx.minimum_spanning_tree(G)

    # Function to extract number for sorting
    def extract_number(bp_name):
        number = extract_bp_number(bp_name)
        return number if number is not None else float('inf')

    # Modified assign_positions function with depth recording
    def assign_positions(T, root_node):
        pos = {}
        parent = {}
        depth_dict = {}
        x_counter = [0]  # Mutable counter to assign unique x positions to leaf nodes

        def recursive_assign(node, depth):
            neighbors = list(T.neighbors(node))
            if depth > 0 and parent[node] in neighbors:
                neighbors.remove(parent[node])

            depth_dict[node] = depth  # Record the depth

            if not neighbors:
                # Leaf node: assign next available x position
                x = x_counter[0]
                pos[node] = (x, -depth)
                x_counter[0] += 1
            else:
                # Internal node: process children and assign x position as the midpoint
                child_x_positions = []
                for neighbor in neighbors:
                    parent[neighbor] = node
                    recursive_assign(neighbor, depth + 1)
                    child_x_positions.append(pos[neighbor][0])

                x = sum(child_x_positions) / len(child_x_positions)
                pos[node] = (x, -depth)

        recursive_assign(root_node, 0)
        return pos, parent, depth_dict

    # Choose a root node
    root_node = min(T.nodes(), key=lambda x: extract_number(x))
    pos, parent, depth_dict = assign_positions(T, root_node)

    # Plot the dendrogram with nodes colored by depth
    def plot_dendrogram(T, pos, depth_dict):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get the maximum depth for normalization
        max_depth = max(depth_dict.values())

        # Choose a colormap
        cmap = plt.get_cmap('viridis')

        # Draw edges with hard angles
        for node in T.nodes():
            parent_node = parent.get(node)
            if parent_node is not None:
                x_parent, y_parent = pos[parent_node]
                x_child, y_child = pos[node]

                # Vertical line from parent to child's level
                ax.plot([x_parent, x_parent], [y_parent, y_child], 'k-')
                # Horizontal line from parent to child
                ax.plot([x_parent, x_child], [y_child, y_child], 'k-')

        # Draw nodes
        node_colors = []
        for node, (x, y) in pos.items():
            depth = depth_dict[node]
            # Normalize depth to [0, 1] for colormap
            depth_norm = depth / max_depth if max_depth > 0 else 0
            color = cmap(depth_norm)
            node_colors.append(color)

            # Use original names for labels
            original_name = reverse_name_mapping.get(node, node)
            ax.scatter(x, y, s=100, c=[color], edgecolors='k')
            ax.text(x, y + 0.1, str(original_name), ha='center', va='bottom', fontsize=8)

        # Create a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_depth))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Depth Level')

        ax.set_title('Duct System Dendrogram Colored by Depth')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    plot_dendrogram(T, pos, depth_dict)

    # Detect communities
    partition = community.best_partition(G)
    # Assign colors based on community
    colors = [partition[node] for node in G.nodes()]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3, node_size=500)
    plt.title('Community Detection in the Duct System')
    plt.show()

    # Calculate centrality
    centrality = nx.betweenness_centrality(G)

    # Normalize centrality values to [0, 1]
    max_centrality = max(centrality.values())
    if max_centrality > 0:
        centrality_norm = {node: val / max_centrality for node, val in centrality.items()}
    else:
        centrality_norm = {node: 0 for node in centrality}

    # Assign colors
    node_colors = [centrality_norm[node] for node in G.nodes()]

    # Create a Figure and Axes object
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the graph on the specified Axes
    nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.viridis, with_labels=True, node_size=500, ax=ax)

    # Create the ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    # Add the colorbar to the same Axes
    plt.colorbar(sm, ax=ax, label='Normalized Betweenness Centrality')

    # Use the node with the highest betweenness centrality as the root
    root_node = max(centrality, key=centrality.get)

    # Reconstruct the tree rooted at the central node
    T = nx.bfs_tree(G, root_node)

    # Assign positions using the dendrogram approach
    pos, parent, depth_dict = assign_positions(T, root_node)
    plot_dendrogram(T, pos, depth_dict)

    # Set the title and show the plot
    ax.set_title('Node Centrality Heatmap')
    plt.show()

# After processing all duct systems, save the cleaned and normalized data back to a JSON file
output_file = 'normalized_annotations.json'
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)
print(f"Normalized data saved to {output_file}")
