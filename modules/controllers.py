import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
if __name__ == '__main__':
    from modules.auxilliary import find_forward_nearest_point_segment, compute_cross_track_error, compute_heading_error
else:
    from modules.auxilliary import find_forward_nearest_point_segment, compute_cross_track_error, compute_heading_error

class PController:
    def __init__(self, kp):
        """Initialize the Proportional Controller.
        Parameters:
        - kp: Proportional gain.
        """
        self.kp = kp
        
    def compute(self, setpoint, state):
        """Compute the control output based on the setpoint and the measured value.
        Parameters:
        - setpoint: Desired target value.
        - state: Current measured value.
        Returns:
        - Control output.
        """
        error = setpoint - state
        return self.kp * error
    
class PIController:
    def __init__(self, kp, ki, dt) -> None:
        self.kp = kp
        self.ki = ki
        self.integral =0.0
        self.dt = dt

    def compute(self, setpoint, state, dt=None) -> float:
        """Compute the control output based on the setpoint and the measured value.
        Parameters:
        - setpoint: Desired target value.
        - state: Current measured value.
        - dt: Time step.
        Returns:
        - Control output.
        """
        dt = self.dt if dt is None else dt
        error = setpoint - state
        self.integral += error * dt
        return self.kp * error + self.ki * self.integral

    def reset(self):
        self.integral = 0.0

class PDController:
    def __init__(self, kp, kd) -> None:
        self.kp = kp
        self.kd = kd
        self.previous_error = 0.0
    
    def compute(self, setpoint, state, dt) -> float:
        """Compute the control output based on the setpoint and the measured value.
        Parameters:
        - setpoint: Desired target value.
        - state: Current measured value.
        - dt: Time step.
        Returns:
        - Control output.
        """
        error = setpoint - state
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.kd * derivative

    def reset(self):
        self.previous_error = 0.0

class PIDController:
    def __init__(self, kp, ki, kd, dt=0.01) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral =0.0
        self.previous_error = 0.0
        self.dt = dt
    
    def compute(self, error, dt=None) -> float:
        """Compute the control output based on the setpoint and the measured value.
        Parameters:
        - setpoint: Desired target value.
        - state: Current measured value.
        - dt: Time step.
        Returns:
        - Control output.
        """
        if dt is None:
            dt = self.dt
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

class StanleyController:
    def __init__(self, k=1.0, v=1.0):
        self.k = k  # Gain for cross-track error correction
        self.v = v  # Velocity of the vehicle

    def control(self, ref_path, current_pose, lookahead_distance=10):
        """
        Computes the steering command to reduce cross-track error.
        ref_path : list of (x, y) tuples
            Reference path.
        current_pose : tuple
            Current pose of the vehicle in the form (x, y, theta).
        """
        x, y, theta = current_pose
        # Find the nearest point on the reference path
        nearest_point = find_forward_nearest_point_segment(x=current_pose[0], y=current_pose[1], theta=current_pose[2], path_x=ref_path[0], path_y=ref_path[1])

        # Compute the cross-track error
        cte = compute_cross_track_error(x=current_pose[0], y=current_pose[1], nearest_x=nearest_point[0], nearest_y=nearest_point[1])

        # Determine the sign of the cross-track error (left or right of the reference path)
        sign = np.sign(np.sin(theta) * (nearest_point[1] - x) - np.cos(theta) * (nearest_point[2] - y))
        heading_error = compute_heading_error(theta=current_pose[2], x=current_pose[0], y=current_pose[1], lookahead_distance=lookahead_distance, path_x=ref_path[0], path_y=ref_path[1])

        # Compute the steering command based on the cross-track and heading errors
        delta = heading_error + np.arctan(self.k * cte * sign / self.v)

        return delta
        

class PurePursuitController:
    def __init__(self, ld=1.6):
        self.ld = ld  # Lookahead distance

    def control(self, ref_path, current_pose):
        """
        Computes the steering command based on the pure pursuit control strategy.
        ref_path : list of (x, y) tuples
            Reference path.
        current_pose : tuple
            Current pose of the vehicle in the form (x, y, theta).
        """
        # Extract current position and orientation
        x, y, theta = current_pose

        # Find the lookahead point by searching for the point on the reference path that's closest to the circle of radius ld around the vehicle
        lookahead_point = None
        min_dist = float('inf')
        for (x_ref, y_ref) in ref_path:
            dist = np.sqrt((x - x_ref)**2 + (y - y_ref)**2)
            if abs(dist - self.ld) < min_dist:
                min_dist = abs(dist - self.ld)
                lookahead_point = (x_ref, y_ref)

        # If no lookahead point is found, choose the last point in the reference path
        if not lookahead_point:
            lookahead_point = ref_path[-1]

        # Compute the steering angle based on the lookahead point
        alpha = np.arctan2(lookahead_point[1] - y, lookahead_point[0] - x) - theta

        # Compute the steering command based on the geometry of the circle that intersects the lookahead point
        delta = np.arctan2(2 * np.sin(alpha), self.ld)

        return delta
    
class MPC:
    def __init__(self, model, u_max, dt, n_horizon, const_vel=True, **kwargs):
        """
        Initialize the MPC class.

        Args:
            model: The system model.
            u_max: The maximum steering angle for bound constraints.
            dt: The time step.
            n_horizon: The number of time steps in the horizon.
            const_vel: A boolean indicating whether the reference velocity is constant.
            **kwargs: Optional keyword arguments.
        """
        self.model = model
        self.x_ref = None
        self.u_max = np.degrees(u_max)  # Maximum steering angle for bound constraints
        self.dt = dt
        self.n_horizon = n_horizon
        self.const_vel = const_vel
        if self.const_vel:
            self.reference_speed = kwargs.get('reference_speed', 1.0)
        self.max_delta_change = kwargs.get('max_delta_change', 5.0)  # Maximum change in steering angle per time step for rate constraints
        self.max_iter = kwargs.get('max_iter', 100)
        self.eps = kwargs.get('eps', 1e-4)
        self.ftol = kwargs.get('ftol', 1e-6)
        self.weight_state = kwargs.get('weight_state', 1.0)
        self.weight_control = kwargs.get('weight_control', 0.1)
        

        # Initialize the state and control variables
        self.x = model.get_state()
        self.u = np.zeros(self.n_horizon)

        self.prev_delta = 0.0

    def predict(self, u):
        """
        Predict the state of the system given the control inputs.

        Args:
            u: The control inputs.

        Returns:
            The total cost.
        """
        initial_state = self.model.get_state()
        x_pred = np.zeros((self.n_horizon, 6))
        x_pred[0] = self.x
        
        for i in range(1, self.n_horizon):
            x_next = self.model.get_state()  # Get the current state
            if not self.const_vel:
                self.model.step(self.x_ref[i][2], u[i-1])  # Here, assuming x_ref has a velocity component
            else:
                self.model.step(self.reference_speed, u[i-1])
            x_pred[i] = self.model.get_state()  # Store the next state
        
        # Calculate state deviation cost
        if not self.const_vel:
            costs = np.sum(np.linalg.norm(x_pred - self.x_ref, axis=1)**2)
        else:
            # costs = np.sum(np.linalg.norm((x_pred[:,0].reshape(self.n_horizon,1), x_pred[:,1].reshape(self.n_horizon,1)) - self.x_ref))
            state_deviation_cost = np.sum(np.linalg.norm(x_pred[:, :2] - self.x_ref[:, :2], axis=1)**2)
        
        # Calculate control effort cost
        control_effort_cost = np.sum(u**2)
        
        total_cost = self.weight_state * state_deviation_cost + self.weight_control * control_effort_cost
            
        self.model.initialize(*initial_state)
        return total_cost

    def optimize(self):
        """
        Optimize the control inputs to minimize the predicted costs.

        Returns:
            The optimized control inputs.
        """        
        x = self.u  # use previous self.u as initial guess for the optimizer (warm start)
        result = minimize(self.predict, x, method='SLSQP', bounds=[(-self.u_max, self.u_max)] * self.n_horizon, options={'maxiter': self.max_iter, 'ftol': self.ftol}, tol=self.eps)
        self.u = result.x  # update self.u with the optimized control inputs
        # Return the optimized control inputs
        return result.x

    def control(self, x_ref):
        """
        Compute and return the optimal control input.

        Args:
            x_ref: The reference state.

        Returns:
            The optimal control input.
        """
        self.x_ref = x_ref
        self.u = self.optimize()
        return self.u[0]
    


#----------------#
# Tuning Methods #
def simulate_stanley(k_values, model, reference_path):
    def tune_stanley(k):
        """
        Simulate the Stanley controller for a given value of k and return the cumulative cross-track error.
        """
        reference_path_x, reference_path_y = zip(*reference_path)
        stanley_controller = StanleyController(k=k)
        cte_cumulative = 0
        for _ in range(len(reference_path_x)):
            nearest_idx, nearest_x, nearest_y = find_forward_nearest_point_segment(model.x, model.y, model.theta, reference_path_x, reference_path_y)
            cte = compute_cross_track_error(model.x, model.y, nearest_x, nearest_y)
            cte_cumulative += abs(cte)
            delta = stanley_controller.control((reference_path_x, reference_path_y), (model.x, model.y, model.theta))
            model.step(1, delta)
        return cte_cumulative
    # Execute the simulation for the Stanley controller
    cte_values = [tune_stanley(k) for k in tqdm(k_values)]

    # Find the k value that minimizes the cumulative cross-track error
    best_k = k_values[np.argmin(cte_values)]
    best_cte = min(cte_values)
    print(f"{best_k=}\n{best_cte=}")
    return best_k, best_cte

def simulate_purepursuit(ld_values, model, reference_path):
    def tune_purepursuit(ld):
        """
        Simulate the PurePursuit controller for a given lookahead distance and return the cumulative cross-track error.
        """
        reference_path_x, reference_path_y = zip(*reference_path)
        purepursuit_controller = PurePursuitController(ld=ld)
        cte_cumulative = 0
        for _ in range(len(reference_path_x)):
            nearest_idx, nearest_x, nearest_y = find_forward_nearest_point_segment(model.x, model.y, model.theta, reference_path_x, reference_path_y)
            cte = compute_cross_track_error(model.x, model.y, nearest_x, nearest_y)
            cte_cumulative += abs(cte)
            delta = purepursuit_controller.control(reference_path, (model.x, model.y, model.theta))
            model.step(1, delta)
        return cte_cumulative

    # Execute the simulation for the PurePursuit controller
    cte_values_purepursuit = [tune_purepursuit(ld) for ld in tqdm(ld_values)]

    # Find the ld value that minimizes the cumulative cross-track error
    best_ld = ld_values[np.argmin(cte_values_purepursuit)]
    print(f"{best_ld=}")
    best_cte_purepursuit = min(cte_values_purepursuit)

    return best_ld, best_cte_purepursuit