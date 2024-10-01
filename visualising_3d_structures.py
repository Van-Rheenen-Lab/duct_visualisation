import json
import numpy as np
import plotly.graph_objects as go


# Function to interpolate z values along the segment
def interpolate_z(start_z, end_z, num_points):
    return np.linspace(start_z, end_z, num_points)


# Load the duct system from a JSON file
def load_duct_system(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data['duct_systems'][0]


# Extract data from the duct system
def extract_coordinates(duct_system):
    branch_points = duct_system['branch_points']
    segments = duct_system['segments']

    x_vals = []
    y_vals = []
    z_vals = []
    branch_x_vals = []
    branch_y_vals = []
    branch_z_vals = []

    # Iterate over the segments and interpolate z values between branch points
    for segment_name, segment_data in segments.items():
        start_bp = branch_points[segment_data['start_bp']]
        end_bp = branch_points[segment_data['end_bp']]

        # Collect start branch point
        branch_x_vals.append(start_bp['x'])
        branch_y_vals.append(start_bp['y'])
        branch_z_vals.append(start_bp['z'])

        # Handle case with no internal points (direct line between start and end points)
        if not segment_data['internal_points']:
            x_vals.append(start_bp['x'])
            y_vals.append(start_bp['y'])
            z_vals.append(start_bp['z'])

            x_vals.append(end_bp['x'])
            y_vals.append(end_bp['y'])
            z_vals.append(end_bp['z'])
        else:
            # Add internal points and interpolate z
            num_internal_points = len(segment_data['internal_points']) + 2  # include start and end points
            z_interp = interpolate_z(start_bp['z'], end_bp['z'], num_internal_points)

            # Add start point
            x_vals.append(start_bp['x'])
            y_vals.append(start_bp['y'])
            z_vals.append(start_bp['z'])

            # Add internal points with interpolated z
            for i, point in enumerate(segment_data['internal_points']):
                x_vals.append(point['x'])
                y_vals.append(point['y'])
                z_vals.append(z_interp[i + 1])

            # Add end point
            x_vals.append(end_bp['x'])
            y_vals.append(end_bp['y'])
            z_vals.append(end_bp['z'])

        # Collect end branch point
        branch_x_vals.append(end_bp['x'])
        branch_y_vals.append(end_bp['y'])
        branch_z_vals.append(end_bp['z'])

    return x_vals, y_vals, z_vals, branch_x_vals, branch_y_vals, branch_z_vals


# Plot the duct system using Plotly
def plot_duct_system(file_path):
    duct_system = load_duct_system(file_path)
    x_vals, y_vals, z_vals, branch_x_vals, branch_y_vals, branch_z_vals = extract_coordinates(duct_system)

    # Create 3D line for duct path
    duct_path = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Duct Path'
    )

    # Create scatter plot for branch points
    branch_points = go.Scatter3d(
        x=branch_x_vals,
        y=branch_y_vals,
        z=branch_z_vals,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Branch Points'
    )

    # Set up the layout
    layout = go.Layout(
        title='3D Visualization of Duct System',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=[duct_path, branch_points], layout=layout)

    # Show the figure
    fig.show()

# Example usage: Pass the path to your JSON file
if __name__ == '__main__':
    file_path = 'test3d.json'
    plot_duct_system(file_path)
