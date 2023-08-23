import numpy as np

from modules.model import KinematicBicycleModel
from modules.controllers import StanleyController, PurePursuitController, simulate_stanley, simulate_purepursuit
from modules.trajectories import total_distance, circle_trajectory, figure_eight_trajectory, double_lane_change_trajectory

def tune(model, controller, reference_path, **kwargs):
    ref_path = list(zip(reference_path[0], reference_path[1]))

    if type(controller) is StanleyController:
        k_values = kwargs.get('k_values', np.linspace(0.001, 1.0, 20))
        best_val, _ = simulate_stanley(k_values=k_values, model=model, reference_path=ref_path)
        return best_val
    elif type(controller) is PurePursuitController:
        ld_values = kwargs.get('ld_values', np.linspace(0.5, 5.0, 20))
        best_val, _ = simulate_purepursuit(ld_values=ld_values, model=model, reference_path=ref_path)
        return best_val

if __name__ == '__main__':
    #-------#
    # Model #
    L = 2.5
    lr = 1.7
    dt = 0.01
    v = 1.0

    kin_model = KinematicBicycleModel(L=L, lr=lr, dt=dt)

    #---------------------------#
    # Reference Path Parameters #
    NUM_POINTS = 1000
    # Circle #
    R = 10  # Radius of Circle
    # Fig 8 #
    A=12  # Diameter of Figure 8
    # Double Lane Change #
    LANE_WIDTH = 3.6
    LANE_CHANGE_DISTANCE = 50
    STRAIGHT_DISTANCE = 1

    circular_gt = circle_trajectory(R=R, num_points=NUM_POINTS)
    simulation_steps = int(total_distance(circular_gt[0], circular_gt[1]) / (v * dt))

    figure8_gt = figure_eight_trajectory(a=A, num_points=NUM_POINTS)
    simulation_steps_fig8 = int(total_distance(figure8_gt[0], figure8_gt[1]) / (v * dt))

    double_gt = double_lane_change_trajectory(lane_width=LANE_WIDTH, lane_change_distance=LANE_CHANGE_DISTANCE, straight_distance=STRAIGHT_DISTANCE, num_points=NUM_POINTS)
    simulation_steps_double = int(total_distance(double_gt[0], double_gt[1]) / (v * dt))
    
    #-------------#
    # Controllers #
    Stanley = StanleyController()
    PurePursuit = PurePursuitController()
    controllers = [PurePursuit, Stanley]

    BEST_VALS = {
        'StanleyController': {'circle': [], 'figure8': [], 'double_lane_change': []},
        'PurePursuitController': {'circle': [], 'figure8': [], 'double_lane_change': []}
    }
    for controller in controllers:
        print(controller.__class__.__name__)
        k_values = np.linspace(0.0001, 1.0, 1000)
        ld_values = np.linspace(0.001, 5.0, 1000)
        try:
            #--------#
            # Circle #
            initial_theta = np.arctan2(circular_gt[1][1] - circular_gt[1][0], circular_gt[0][1] - circular_gt[0][0])

            kin_model.initialize(x=circular_gt[0][0], y=circular_gt[1][0], theta=initial_theta)
            best_val = tune(model=kin_model, controller=controller, reference_path=circular_gt, k_values=k_values, ld_values=ld_values)
            BEST_VALS[controller.__class__.__name__]["circle"].append(best_val)
            kin_model.reset()

            #----------#
            # Figure 8 #
            initial_theta = np.arctan2(figure8_gt[1][1] - figure8_gt[1][0], figure8_gt[0][1] - figure8_gt[0][0])

            kin_model.initialize(x=figure8_gt[0][0], y=figure8_gt[1][0], theta=initial_theta)
            best_val = tune(model=kin_model, controller=controller, reference_path=figure8_gt, k_values=k_values, ld_values=ld_values)
            BEST_VALS[controller.__class__.__name__]["figure8"].append(best_val)
            kin_model.reset()

            #--------------------#
            # Double Lane Change #
            kin_model.initialize(x=double_gt[0][0], y=double_gt[1][0])
            best_val = tune(model=kin_model, controller=controller, reference_path=double_gt, k_values=k_values, ld_values=ld_values)
            BEST_VALS[controller.__class__.__name__]["double_lane_change"].append(best_val)
            kin_model.reset()
        except Exception as e:
            print(e)
            continue
    print(BEST_VALS)