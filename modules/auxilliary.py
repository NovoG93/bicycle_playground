import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

def find_forward_nearest_point_segment(x, y, theta, path_x, path_y, segment_length=20):
    """
    Find the nearest point on the path that's ahead of the vehicle's current position using a segment-based approach.

    Parameters:
    - x, y: Current position of the vehicle.
    - theta: Current heading direction of the vehicle.
    - path_x, path_y: X and Y coordinates of the path.
    - segment_length: Number of points to consider around the nearest point.

    Returns:
    - Index, x, and y coordinates of the nearest point ahead on the path.
    """
    # Initially find the nearest point using the original method
    nearest_idx_initial, _, _ = find_nearest_point(x, y, path_x, path_y)
    
    # Determine the segment of points to consider
    start_idx = max(0, nearest_idx_initial - segment_length)
    end_idx = min(len(path_x), nearest_idx_initial + segment_length + 1)
    
    # Direction vector of the vehicle
    vehicle_dir = np.array([np.cos(theta), np.sin(theta)])
    
    forward_points_idx = []
    for i in range(start_idx, end_idx):
        # Direction vector towards the path point
        point_dir = np.array([path_x[i] - x, path_y[i] - y])
        # Check if the point lies in the same direction as the vehicle's movement
        if np.dot(vehicle_dir, point_dir) > 0:  # dot product > 0 means the angle is < 90 degrees
            forward_points_idx.append(i)
    
    if not forward_points_idx:  # If no forward points found, just use the initial nearest point
        return nearest_idx_initial, path_x[nearest_idx_initial], path_y[nearest_idx_initial]
    
    distances_sq = [(path_x[i] - x)**2 + (path_y[i] - y)**2 for i in forward_points_idx]
    min_distance_idx = np.argmin(distances_sq)
    nearest_idx = forward_points_idx[min_distance_idx]
    nearest_x = path_x[nearest_idx]
    nearest_y = path_y[nearest_idx]
    
    return nearest_idx, nearest_x, nearest_y

def find_nearest_point(x: float, y: float, path_x: List, path_y: List) -> Tuple[int, float, float]:
    """
    Find the nearest point on a given path to the current vehicle position.
    
    Parameters:
    - x (float): Current x-coordinate of the vehicle.
    - y (float): Current y-coordinate of the vehicle.
    - path_x (array): X-coordinates of the reference path.
    - path_y (array): Y-coordinates of the reference path.
    
    Returns:
    - nearest_idx (int): Index of the nearest point on the path.
    - nearest_x (float): X-coordinate of the nearest point.
    - nearest_y (float): Y-coordinate of the nearest point.
    """
    
    # Compute the squared distances to each point on the path
    distances_sq = (path_x - x)**2 + (path_y - y)**2
    
    # Find the index of the minimum distance (nearest point)
    nearest_idx = np.argmin(distances_sq)
    nearest_x = path_x[nearest_idx]
    nearest_y = path_y[nearest_idx]
    
    return nearest_idx, nearest_x, nearest_y

def visualize_nearest_points(reference_path_x, reference_path_y, driven_path_x, driven_path_y):
    """
    Visualize the nearest points on the reference path corresponding to each point on the driven path.

    Parameters:
    - reference_path_x, reference_path_y: X and Y coordinates of the reference path.
    - driven_path_x, driven_path_y: X and Y coordinates of the driven path.
    """
    # Find the nearest points on the reference path for each point on the driven path
    nearest_points_x = []
    nearest_points_y = []
    for x, y in zip(driven_path_x, driven_path_y):
        _, nearest_x, nearest_y = find_nearest_point(x, y, reference_path_x, reference_path_y)
        nearest_points_x.append(nearest_x)
        nearest_points_y.append(nearest_y)

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(reference_path_x, reference_path_y, 'k-', label="Reference Path", linewidth=2)
    plt.scatter(driven_path_x, driven_path_y, color='blue', label="Driven Path", s=50)
    plt.scatter(nearest_points_x, nearest_points_y, color='red', label="Nearest Points on Reference Path", s=50, marker='x')
    # Add arrows from driven path to the corresponding nearest point on the reference path
    for dx, dy, nx, ny in zip(driven_path_x, driven_path_y, nearest_points_x, nearest_points_y):
        plt.arrow(dx, dy, nx-dx, ny-dy, color='gray', head_width=0.2, head_length=0.4, length_includes_head=True)
    
    plt.title("Driven Path and Corresponding Nearest Points on Reference Path")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def compute_cross_track_error(x, y, nearest_x, nearest_y):
    """
    Compute the Cross Track Error (CTE) given the vehicle position and nearest point on the path.
    
    Parameters:
    - x (float): Current x-coordinate of the vehicle.
    - y (float): Current y-coordinate of the vehicle.
    - nearest_x (float): X-coordinate of the nearest point on the path.
    - nearest_y (float): Y-coordinate of the nearest point on the path.
    
    Returns:
    - cte (float): Cross Track Error.
    """
    
    # CTE is simply the Euclidean distance from the vehicle to the nearest point
    cte = np.sqrt((x - nearest_x)**2 + (y - nearest_y)**2)
    
    return cte

def compute_heading_error(theta: float, x: float, y: float, lookahead_distance: int,
                          path_x, path_y) -> float:
    """
    Compute the Heading Error given the vehicle's heading and position, and the path.
    
    Parameters:
    - theta: Current heading angle of the vehicle (in radians).
    - x, y: Current position of the vehicle.
    - lookahead_distance: Number of points to consider for averaging.
    - path_x, path_y: Coordinates of the reference path.
    
    Returns:
    - heading_error: Normalized heading error in radians.
    """
    nearest_idx, _, _ = find_forward_nearest_point_segment(x, y, theta, path_x, path_y)
    if nearest_idx + lookahead_distance < len(path_x):
        lookahead_x = path_x[nearest_idx + lookahead_distance]
        lookahead_y = path_y[nearest_idx + lookahead_distance]
    else:
        lookahead_x = path_x[-1]
        lookahead_y = path_y[-1]
    
    desired_heading = np.arctan2(lookahead_y - y, lookahead_x - x)
    heading_error = desired_heading - theta
    # Normalize the heading error to [-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    return heading_error