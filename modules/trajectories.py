import numpy as np

if __name__ == '__main__':
    from model import KinematicBicycleModel
    from controllers import PController, PIController, PIDController
else:
    from modules.model import KinematicBicycleModel
    from modules.controllers import PController, PIController, PIDController

def add_landmarks(ref_x, ref_y, num):
    """Adds landmarks along the trajectory and returns a list of tuples(id, x, y)"""
    # Add landmarks along the path
    landmarks = []
    for i in range(num):
        # Pick a random index along the path
        idx = np.random.randint(len(ref_x))
        landmarks.append((i, ref_x[idx], ref_y[idx]))
    return landmarks

def total_distance(x, y):
    """Compute the total distance for a trajectory defined by x and y coordinates."""
    # Compute the incremental distances between consecutive points
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(distances)

def circle_trajectory(R, num_points=1000, segment_angle=360.0):
    """
    Generate a circular path or segment based on the given radius.
    
    Parameters:
    - R (float): Desired radius of the circle.
    - num_points (int): Number of points in the circular path. Default is 1000.
    - segment_angle (float): Angle of the circular segment in degrees. Default is 360 (complete circle).
    
    Returns:
    - x (array): X-coordinates of the circular path.
    - y (array): Y-coordinates of the circular path.
    """
    
    # Calculate the angle step for each iteration
    dtheta = np.deg2rad(segment_angle) / num_points
    
    # Compute the circular path
    theta_values = np.arange(0, np.deg2rad(segment_angle) + dtheta, dtheta)
    x = R + R * np.cos(theta_values)
    y = R * np.sin(theta_values)
    
    return x, y


def figure_eight_trajectory(a=8.0, num_points=500):
    """
    Generate a figure-eight path in a specific sequence.
    
    Parameters:
        a (float): Scaling factor.
        num_points (int): Number of points in the path.
        
    Returns:
        (tuple): Two lists representing the X and Y coordinates of the path.
    """
    """
    Generate a figure-eight path in the specified sequence.
    
    Parameters:
        a (float): Scaling factor.
        num_points (int): Number of points in the path.
        
    Returns:
        list of tuples: Representing the (X, Y) coordinates of the path.
    """
    adjusted_num_points = num_points // 4
    right_upper_t = np.flip(np.linspace(np.pi, 3*np.pi/2, adjusted_num_points, endpoint=False))
    right_lower_t = np.flip(np.linspace(3*np.pi/2, 2*np.pi, adjusted_num_points, endpoint=False))
    left_upper_t = np.flip(np.linspace(0, np.pi/2, adjusted_num_points, endpoint=False))
    left_lower_t = np.flip(np.linspace(np.pi/2, np.pi, adjusted_num_points, endpoint=True))
    x_trajectory = []
    y_trajectory = []
    for t_segment in [left_upper_t, right_lower_t, right_upper_t, left_lower_t]:
        for t in t_segment:
            denominator = np.sin(t)**2 + 1
            x = a * np.sqrt(2) * np.cos(t) / denominator
            y = a * np.sqrt(2) * np.sin(t) * np.cos(t) / denominator
            x_trajectory.append(x)
            y_trajectory.append(y)
    return x_trajectory, y_trajectory

def double_lane_change_trajectory(lane_width=3.6, lane_change_distance=50, straight_distance=19, num_points=1000):
    def quintic_polynomial(t):
        """Generate the quintic polynomial for the lane change."""
        a_0 = 0
        a_1 = 0
        a_2 = 0
        a_3 = 10
        a_4 = -15
        a_5 = 6
        return a_0 + a_1*t + a_2*t**2 + a_3*t**3 + a_4*t**4 + a_5*t**5

    # Total distance for the maneuver
    total_distance = 2 * lane_change_distance + 2 * 50 + straight_distance  # two lane changes and straight_distancem straight drive
    
    # Create an array of distances for which we'll calculate lateral displacements
    distances = np.linspace(0, total_distance, num_points)
    
    # Calculate the time (or equivalent distance) required for one lane change
    lane_change_duration = 50  # This can be adjusted
    
    # Initialize an array for lateral displacements
    y = np.zeros_like(distances)
    
    # Calculate lane change using quintic polynomial
    for i, d in enumerate(distances):
        # First lane change
        if lane_change_distance <= d < lane_change_distance + lane_change_duration:
            t = (d - lane_change_distance) / lane_change_duration
            y[i] = lane_width * quintic_polynomial(t)
        
        # Driving straight after first lane change
        elif lane_change_distance + lane_change_duration <= d < lane_change_distance + lane_change_duration + straight_distance:
            y[i] = lane_width
        
        # Second lane change
        elif lane_change_distance + lane_change_duration + straight_distance <= d < lane_change_distance + 2 * lane_change_duration + straight_distance:
            t = (d - lane_change_distance - lane_change_duration - straight_distance) / lane_change_duration
            y[i] = lane_width * (1 - quintic_polynomial(t))
        
        # Driving straight after second lane change
        elif d >= lane_change_distance + 2 * lane_change_duration + straight_distance:
            y[i] = 0
        
    # Return the trajectory as a list of tuples (x,y)
    return (distances, y)