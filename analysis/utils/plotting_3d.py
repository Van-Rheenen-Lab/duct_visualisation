import networkx as nx
from .plotting_trees import get_segment_color

def plot_3d_system(G, system_data, annotation_to_color):
    import plotly.graph_objects as go
    import plotly.express as px

    color_scale = px.colors.qualitative.Plotly
    connected_components = list(nx.connected_components(G.to_undirected()))

    fig = go.Figure()

    # Plot edges per connected component
    for i, component in enumerate(connected_components):
        for u, v, data in G.edges(data=True):
            if u in component and v in component:
                x1 = system_data["branch_points"][u]["x"]
                y1 = system_data["branch_points"][u]["y"]
                z1 = system_data["branch_points"][u]["z"]
                x2 = system_data["branch_points"][v]["x"]
                y2 = system_data["branch_points"][v]["y"]
                z2 = system_data["branch_points"][v]["z"]

                segment_name = data.get('segment_name', None)
                c = 'black'
                if segment_name and 'segments' in system_data:
                    segment_data = system_data['segments'].get(segment_name, None)
                    if segment_data:
                        c = get_segment_color(segment_data, annotation_to_color)

                fig.add_trace(
                    go.Scatter3d(
                        x=[x1, x2],
                        y=[y1, y2],
                        z=[z1, z2],
                        mode='lines',
                        # set size to 0 to avoid markers at the end of the line
                        marker=dict(size=0, color=c),
                        line=dict(color=c, width=4),
                        name= segment_data['properties'].get('Annotation', segment_name) if segment_data else segment_name
                    )
                )


    fig.update_layout(scene=dict(aspectmode='cube'), title="3D Plot of Duct System")

    # show labels
    for node, data in system_data["branch_points"].items():
        x = data["x"]
        y = data["y"]
        z = data["z"]
        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers',
                marker=dict(size=0, color='red'),
                text=node,
                name="Branch Point",
                showlegend=False
            )
        )

    fig.show()