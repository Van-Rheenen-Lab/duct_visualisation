import networkx as nx
from .plotting_trees import get_segment_color

def plot_3d_system(G, annotation_to_color):
    import plotly.graph_objects as go
    import plotly.express as px

    # Use Plotly's qualitative color scale (if needed)
    color_scale = px.colors.qualitative.Plotly
    connected_components = list(nx.connected_components(G.to_undirected()))

    fig = go.Figure()

    # Loop over connected components and edges
    for component in connected_components:
        for u, v, data in G.edges(data=True):
            if u in component and v in component:
                # Extract branch point coordinates directly from graph nodes.
                x1 = G.nodes[u].get("x")
                y1 = G.nodes[u].get("y")
                z1 = G.nodes[u].get("z", 0)  # default to 0 if not provided
                x2 = G.nodes[v].get("x")
                y2 = G.nodes[v].get("y")
                z2 = G.nodes[v].get("z", 0)

                # Use the stored segment name or default to "u_to_v"
                segment_name = data.get("segment_name", f"{u}_to_{v}")
                # Determine the color using edge properties.
                c = get_segment_color(data, annotation_to_color)

                # Use the edge's annotation (if any) as the trace name.
                annotation_label = data.get("properties", {}).get("Annotation", segment_name)

                fig.add_trace(
                    go.Scatter3d(
                        x=[x1, x2],
                        y=[y1, y2],
                        z=[z1, z2],
                        mode="lines",
                        marker=dict(size=0, color=c),
                        line=dict(color=c, width=4),
                        name=annotation_label
                    )
                )

    fig.update_layout(scene=dict(aspectmode="cube"), title="3D Plot of Duct System")

    # Plot branch points with labels
    for node, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        z = data.get("z", 0)
        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers+text",
                marker=dict(size=4, color="red"),
                text=[str(node)],
                name="Branch Point",
                showlegend=False
            )
        )

    fig.show()
