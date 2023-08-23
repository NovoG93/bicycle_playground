# Bicycle Playground for Vehicle Controllers

This repository contains implementations of various controllers for autonomous vehicles, based on the kinematic bicycle model.

## Kinematic Bicycle Model

The kinematic bicycle model is a simplified representation of a vehicle's motion. It is called a "bicycle model" because it assumes the vehicle has two wheels, like a bicycle, even though it can represent a car. The kinematic bicycle model is concerned with the geometry of the vehicle's motion and not the forces acting on it.

The state of the vehicle is described by the following variables:

- $ x, y $: Position of the vehicle's center of gravity (CG).
- $ \theta $: Heading angle.
- $ v $: Velocity.
- $ \delta $: Steering angle.
- $ \beta $: Slip angle, which is the angle between the vehicle's heading and its velocity vector.

The equations of motion for the kinematic bicycle model are:

$$
\begin{align*}
\dot{x} &= v \cos(\theta + \beta) \\
\dot{y} &= v \sin(\theta + \beta) \\
\dot{\theta} &= \frac{v}{L} \tan(\delta) \\
\beta &= \arctan\left(\frac{l_r \tan(\delta)}{L}\right)
\end{align*}
$$

Where:
- $ L $ is the wheelbase.
- $ l_r $ is the distance from the rear wheel to the center of gravity (CG).

## Controllers Implemented

* __P Controller:__ A simple controller that adjusts the control output linearly based on the error
* __PI Controller:__ Extends the P controller by also considering the integral of the error over time, which helps eliminate steady-state errors
* __PID Controller:__ Extends the PI controller by also considering the rate of change of the error. This provides a damping effect, making the system more responsive and stable
* __Stanley Controller:__ Designed specifically for trajectory tracking problems in autonomous vehicles. It combines cross-track error with heading error to compute the control command
* __Pure Pursuit Controller:__ A geometric controller that looks ahead of the current position to compute control commands
* __Model Predictive Controller:__ An optimization-based controller that predicts the future states of the system over a finite horizon and selects the control inputs that minimize a cost function


## Auxiliary Methods

The repository also contains auxiliary methods for:

- Computing cross-track and heading errors.
- Finding the nearest points on the reference trajectory.
- Visualizing the performance of controllers.

## Usage

Detailed usage instructions and examples are provided in the [Jupyter notebook](./kinematic_model.ipynb).
