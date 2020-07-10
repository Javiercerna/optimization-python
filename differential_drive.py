'''
Adapted from this repository:
https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi/tree/master/workshop_github
'''

import casadi
import matplotlib.pyplot as plt
import numpy as np
import yaml


class NonLinearMPC(object):
    def __init__(self, controller_parameters):
        ## Prediction horizon (in steps)
        self.N = controller_parameters['N']
        ## Weights on the inputs (speed, angular_speed) for all horizon steps
        self.R = casadi.DM([[controller_parameters['R_speed'], 0],
                            [0, controller_parameters['R_angular_speed']]])
        ## Weight on the total time
        self.kappa = controller_parameters['kappa']
        ## Number of states (x, y angle)
        self.nx = 3
        ## Number of inputs (speed, angular_speed)
        self.nu = 2
        ## Minimum speed (in meters/seconds)
        self.speed_min = controller_parameters['speed_min']
        ## Maximum speed (in meters/seconds)
        self.speed_max = controller_parameters['speed_max']
        ## Minimum angular speed (in radians/seconds)
        self.angular_speed_min = controller_parameters['angular_speed_min']
        ## Maximum angular speed (in radians/seconds)
        self.angular_speed_max = controller_parameters['angular_speed_max']
        ## Casadi solver
        self.solver = casadi.Opti()
        ## Casadi states X = [x, y, angle]
        self.X = None
        ## Casadi inputs U = [speed, angular_speed]
        self.U = None
        ## Casadi total time T
        self.T = None
        ## Casadi initial state X_0 = [x_0, y_0, angle_0]
        self.initial_state = None
        ## Casadi goal state X_G = [x_G, y_G, angle_G]
        self.goal_state = None

    def setup_optimization_problem(self):
        self.X = self.solver.variable(self.nx, self.N + 1)
        self.U = self.solver.variable(self.nu, self.N)
        self.T = self.solver.variable()

        self.initial_state = self.solver.parameter(self.nx)
        self.goal_state = self.solver.parameter(self.nx)

        self._add_equality_constraints()
        self._add_inequality_constraints()
        self._add_cost_function()

        self.solver.solver('ipopt')

    def _add_equality_constraints(self):
        self.solver.subject_to(self.X[:, 0] == self.initial_state)
        self.solver.subject_to(self.X[:, -1] == self.goal_state)

        dt = self.T / self.N

        for k in range(self.N):
            self.solver.subject_to(
                [self.X[0, k + 1] == self.X[0, k] + dt * self.U
                 [0, k] * casadi.cos(self.X[2, k]),
                 self.X[1, k + 1] == self.X[1, k] + dt * self.U
                 [0, k] * casadi.sin(self.X[2, k]),
                 self.X[2, k + 1] == self.X[2, k] + dt * self.U
                 [1, k]
                 ]
            )

    def _add_inequality_constraints(self):
        self.solver.subject_to(self.T >= 0)

        self.solver.subject_to(self.speed_min <= self.U[0, :])
        self.solver.subject_to(self.U[0, :] <= self.speed_max)

        self.solver.subject_to(self.angular_speed_min <= self.U[1, :])
        self.solver.subject_to(self.U[1, :] <= self.angular_speed_max)

    def _add_cost_function(self):
        sum_input_costs = 0

        for k in range(self.N):
            sum_input_costs += self.U[:, k].T @ self.R @ self.U[:, k]

        self.solver.minimize(sum_input_costs + self.kappa * self.T)

    def compute_optimal_trajectory(self, initial_state, goal_state):
        self.solver.set_value(self.initial_state, initial_state)
        self.solver.set_value(self.goal_state, goal_state)

        result = self.solver.solve()

        return result.value(self.X)


if __name__ == '__main__':
    with open('differential_drive.yaml', 'r') as f:
        controller_parameters = yaml.load(f, Loader=yaml.SafeLoader)

    non_linear_mpc = NonLinearMPC(controller_parameters)

    non_linear_mpc.setup_optimization_problem()

    initial_state = [0, 0, 0]
    goal_state = [5, 5, 0]

    trajectory = non_linear_mpc.compute_optimal_trajectory(
        initial_state, goal_state
    )

    x = np.array(trajectory[0, :])
    y = np.array(trajectory[1, :])

    plt.plot(initial_state[0], initial_state[1], 'bx')
    plt.plot(goal_state[0], goal_state[1], 'gx')
    plt.plot(x, y)

    plt.show()
