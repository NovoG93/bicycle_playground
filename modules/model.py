import numpy as np

class KinematicBicycleModel:
    """
    Kinematic Bicycle Model class.
    This model represents the kinematic behavior of the vehicle, which means it's primarily concerned with geometry and not the forces or dynamics.
    It assumes that the vehicle can change its velocity instantaneously, so it directly requires the new velocity v as an input to compute the next state.
    """
    def __init__(self, L: float = 2.0, lr: float = 1.2, dt: float = 0.01):
        self.L: float = L  # Wheelbase
        self.lr: float = lr  # Distance from rear wheel to CG
        self.dt: float = dt  # Time step
        self.v: float = 0.0  # Initial velocity
        self.x: float = 0.0  # Initial position x
        self.y: float = 0.0  # Initial position y
        self.theta: float = 0.0  # Heading angle
        self.beta: float = 0.0  # Slip angle
        self.delta: float = 0.0  # Steering angle
        self.max_delta: float = np.radians(30)  # Maximum steering angle

    def initialize(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0, v: float = 0.0, delta: float = 0.0, beta: float = 0.0) -> None:
        """Initialize the state of the vehicle."""
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.delta = delta
        self.beta = beta

    def get_state(self) -> tuple[float, float, float, float, float, float]:
        """Return the current state of the vehicle."""
        return (self.x, self.y, self.theta, self.v, self.delta, self.beta)

    def step(self, v: float, delta: float) -> None:
        """Simulate the kinematic bicycle model for one time step."""
        self.x += v * np.cos(self.theta + self.beta) * self.dt
        self.y += v * np.sin(self.theta + self.beta) * self.dt
        self.theta += (v / self.L) * np.tan(delta) * self.dt
        self.v = v
        self.delta = delta
        self.beta = np.arctan(self.lr * np.tan(delta) / self.L)

    def reset(self) -> None:
        """Reset the state of the vehicle."""
        self.v = 0.0
        self.x = 0.0
        self.theta = 0.0
        self.y = 0.0
        self.beta = 0.0
        self.delta = 0.0