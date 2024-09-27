import json
import networkx as nx
import matplotlib.pyplot as plt
import community  # Install with 'pip install python-louvain'

# Load the JSON data
with open(r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\annotations_Hristina_first try.json', 'r') as file:
    data = json.load(file)

branch_points = data['branch_points']
segments = data['segments']

# Create the graph
G = nx.Graph()
for bp_name in branch_points.keys():
    G.add_node(bp_name)
for segment_data in segments.values():
    G.add_edge(segment_data['start_bp'], segment_data['end_bp'])

# Ensure the graph is a tree
if nx.is_tree(G):
    T = G
else:
    T = nx.minimum_spanning_tree(G)

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
root_node = min(T.nodes(), key=lambda x: int(x))
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

        ax.scatter(x, y, s=100, c=[color], edgecolors='k')
        ax.text(x, y + 0.1, str(node), ha='center', va='bottom', fontsize=8)

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_depth))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Depth Level')

    ax.set_title('Duct System Dendrogram Colored by Depth')
    ax.axis('off')
    plt.tight_layout()

plot_dendrogram(T, pos, depth_dict)

# Detect communities
partition = community.best_partition(G)
# Assign colors based on community
colors = [partition[node] for node in G.nodes()]

plt.figure(figsize=(12, 8))
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set3, node_size=500)
plt.title('Community Detection in the Duct System')


# Calculate centrality
centrality = nx.betweenness_centrality(G)

# Normalize centrality values to [0, 1]
max_centrality = max(centrality.values())
centrality_norm = {node: val / max_centrality for node, val in centrality.items()}

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
