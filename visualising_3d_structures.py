import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import sys
import traceback
import imageio

# Function to interpolate z values along the segment
def interpolate_z(start_z, end_z, num_points):
    return np.linspace(start_z, end_z, num_points)

# Load the duct system from a JSON file
def load_duct_system(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if 'duct_systems' not in data or not data['duct_systems']:
            raise ValueError("JSON does not contain 'duct_systems' or it's empty.")
        return data['duct_systems'][0]
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

# Extract data from the duct system
def extract_coordinates(duct_system):
    try:
        branch_points = duct_system['branch_points']
        segments = duct_system['segments']

        x_vals = []
        y_vals = []
        z_vals = []
        branch_x_vals = []
        branch_y_vals = []
        branch_z_vals = []

        for segment_name, segment_data in segments.items():
            start_bp_id = segment_data.get('start_bp')
            end_bp_id = segment_data.get('end_bp')

            if start_bp_id not in branch_points or end_bp_id not in branch_points:
                print(f"Segment '{segment_name}' has invalid branch point IDs.")
                continue

            start_bp = branch_points[start_bp_id]
            end_bp = branch_points[end_bp_id]

            # Collect start branch point
            branch_x_vals.append(start_bp['x'])
            branch_y_vals.append(start_bp['y'])
            branch_z_vals.append(start_bp['z'])

            internal_points = segment_data.get('internal_points', [])
            num_internal = len(internal_points)
            num_total_points = num_internal + 2  # Start and end points

            if num_total_points < 2:
                print(f"Segment '{segment_name}' has insufficient points.")
                continue

            # Interpolate z values
            z_interp = interpolate_z(start_bp['z'], end_bp['z'], num_total_points)

            # Add start point
            x_vals.append(start_bp['x'])
            y_vals.append(start_bp['y'])
            z_vals.append(start_bp['z'])

            # Add internal points
            for i, point in enumerate(internal_points):
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

    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        traceback.print_exc()
        sys.exit(1)

# Plot the duct system using Matplotlib
def plot_duct_system(x_vals, y_vals, z_vals, branch_x_vals, branch_y_vals, branch_z_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot duct paths
    ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=2, label='Duct Path')

    # Plot branch points
    ax.scatter(branch_x_vals, branch_y_vals, branch_z_vals, color='red', s=50, label='Branch Points')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Duct System')

    ax.legend()

    return fig, ax

# Animate the rotation of the 3D plot
def animate_rotation(fig, ax, output_gif='duct_system.gif', output_mp4='duct_system.mp4', frames=60, interval=50):
    try:
        def update(angle):
            ax.view_init(elev=30, azim=angle)
            return fig,

        ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 360, frames), interval=interval, blit=False)

        # Save as GIF
        ani.save(output_gif, writer='imagemagick')
        print(f"Successfully created GIF: {output_gif}")

        # Save as MP4
        ani.save(output_mp4, writer='ffmpeg')
        print(f"Successfully created MP4 video: {output_mp4}")

    except Exception as e:
        print(f"Error during animation: {e}")
        traceback.print_exc()
        sys.exit(1)

# Alternative: Save frames and create GIF manually if animation fails
def save_frames(fig, ax, output_dir='frames', frames=60):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Output directory already exists: {output_dir}")

        for i, angle in enumerate(np.linspace(0, 360, frames)):
            ax.view_init(elev=30, azim=angle)
            filename = os.path.join(output_dir, f'frame_{i:03d}.png')
            plt.savefig(filename)
            print(f"Saved frame {i + 1}/{frames}: {filename}")

    except Exception as e:
        print(f"Error saving frames: {e}")
        traceback.print_exc()
        sys.exit(1)

# Assemble frames into a GIF
def create_gif_from_frames(output_dir='frames', gif_name='duct_system.gif', fps=20):
    try:
        file_names = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png')])
        if not file_names:
            print("No frames found to create GIF.")
            return

        images = []
        for filename in file_names:
            file_path = os.path.join(output_dir, filename)
            images.append(imageio.imread(file_path))

        imageio.mimsave(gif_name, images, fps=fps)
        print(f"Successfully created GIF: {gif_name}")

    except Exception as e:
        print(f"Error creating GIF: {e}")
        traceback.print_exc()
        sys.exit(1)

# Assemble frames into a video
def create_video_from_frames(output_dir='frames', video_name='duct_system.mp4', fps=20):
    try:
        file_names = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png')])
        if not file_names:
            print("No frames found to create video.")
            return

        images = []
        for filename in file_names:
            file_path = os.path.join(output_dir, filename)
            images.append(imageio.imread(file_path))

        imageio.mimsave(video_name, images, fps=fps)
        print(f"Successfully created MP4 video: {video_name}")

    except Exception as e:
        print(f"Error creating video: {e}")
        traceback.print_exc()
        sys.exit(1)

# Main function to orchestrate the workflow
def main():
    try:
        # Configuration
        file_path = 'test3d.json'        # Path to your JSON file
        output_gif = 'duct_system.gif'   # Output GIF name
        output_mp4 = 'duct_system.mp4'   # Output MP4 video name
        num_frames = 60                   # Number of frames for full rotation
        interval = 50                     # Interval between frames in milliseconds

        # Load and process data
        print("Loading duct system data...")
        duct_system = load_duct_system(file_path)
        print("Extracting coordinates...")
        x_vals, y_vals, z_vals, branch_x_vals, branch_y_vals, branch_z_vals = extract_coordinates(duct_system)

        # Plot duct system
        print("Plotting duct system...")
        fig, ax = plot_duct_system(x_vals, y_vals, z_vals, branch_x_vals, branch_y_vals, branch_z_vals)

        # Animate rotation
        print("Animating rotation and saving outputs...")
        animate_rotation(fig, ax, output_gif=output_gif, output_mp4=output_mp4, frames=num_frames, interval=interval)

        print("All tasks completed successfully.")

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
