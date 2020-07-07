'''
Adapted from this repository:
https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi/tree/master/workshop_github
'''

import casadi
import matplotlib.pyplot as plt
import numpy as np
import yaml

import math


class NonLinearMPC(object):
    def __init__(self, controller_parameters):
        ## Prediction horizon (in steps)
        self.N = controller_parameters['N']
        ## Sampling time (in seconds)
        self.dt = controller_parameters['dt']
        ## Weights on the state (x, y, angle) for all horizon steps
        self.Q = casadi.SX([[controller_parameters['Q_x'], 0, 0],
                            [0, controller_parameters['Q_y'], 0],
                            [0, 0, controller_parameters['Q_angle']]])
        ## Weights on the inputs (speed, angular_speed) for all horizon steps
        self.R = casadi.SX([[controller_parameters['R_speed'], 0],
                            [0, controller_parameters['R_angular_speed']]])
        ## Number of states (x, y angle)
        self.nx = 3
        ## Number of inputs (speed, angular_speed)
        self.nu = 2
        ## Constraints in the state
        self.position_limits = [
            controller_parameters['position_min'],
            controller_parameters['position_max']
        ]
        ## Constraints in the inputs
        self.speed_limits = [
            controller_parameters['speed_min'],
            controller_parameters['speed_max']
        ]
        self.angular_speed_limits = [
            controller_parameters['angular_speed_min'],
            controller_parameters['angular_speed_max']
        ]
        ## Casadi solver
        self.solver = None
        ## Casadi lower bounds on the state (inputs "u")
        self.lower_bound_x = [self.speed_limits[0], self.angular_speed_limits[0]] * self.N  # noqa
        ## Casadi upper bounds on the state (inputs "u")
        self.upper_bound_x = [self.speed_limits[1], self.angular_speed_limits[1]] * self.N  # noqa
        ## Casadi function to reconstruct the trajectory from the inputs
        self.reconstruct_trajectory = None

    def setup_optimization_problem(self):
        x = casadi.SX.sym('x')
        y = casadi.SX.sym('y')
        angle = casadi.SX.sym('angle')
        speed = casadi.SX.sym('speed')
        angular_speed = casadi.SX.sym('angular_speed')

        states = [x, y, angle]
        controls = [speed, angular_speed]

        control_action = [
            self.dt * speed * casadi.cos(angle),
            self.dt * speed * casadi.sin(angle),
            self.dt * angular_speed
        ]

        f = casadi.Function(
            'f',
            states + controls, control_action,
            ['x', 'y', 'angle', 'speed', 'angular_speed'],
            ['d_x', 'd_y', 'd_angle']
        )

        U = casadi.SX.sym('U', self.nu, self.N)
        P = casadi.SX.sym('P', 2 * self.nx)  # Initial state + reference state

        X = casadi.SX.sym('X', self.nx, self.N + 1)

        X[:, 0] = P[0:self.nx]

        for k in range(self.N):
            f_values = f(
                x=X[:, k][0], y=X[:, k][1], angle=X[:, k][2],
                speed=U[:, k][0], angular_speed=U[:, k][1]
            )
            X[:, k + 1] = X[:, k] + casadi.vertcat(*f_values.values())

        J = self._make_cost_function(X, U, P)
        constraints = self._make_constraints(X)

        nlp = {
            'x': casadi.reshape(U, 2 * self.N, 1),
            'f': J,
            'g': casadi.vertcat(*constraints),
            'p': P
        }

        self.solver = casadi.nlpsol('S', 'ipopt', nlp)
        self.reconstruct_trajectory = self._make_reconstruct_trajectory(
            X, U, P
        )

    def compute_optimal_trajectory(self, initial_state, goal_state):
        initial_inputs = np.zeros((self.N, 2))

        result = self.solver(
            x0=casadi.reshape(initial_inputs.T, 2 * self.N, 1),
            lbg=self.position_limits[0],
            ubg=self.position_limits[1],
            lbx=self.lower_bound_x,
            ubx=self.upper_bound_x,
            p=initial_state + goal_state
        )

        trajectory = self.reconstruct_trajectory(
            casadi.vertcat(result['x'], initial_state, goal_state)
        )

        return trajectory

    def _make_cost_function(self, X, U, P):
        J = 0

        for k in range(self.N):
            state_objective = X[:, k] - P[self.nx:]
            state_cost = state_objective.T @ self.Q @ state_objective
            input_cost = U[:, k].T @ self.R @ U[:, k]
            J += state_cost + input_cost

        return J

    def _make_constraints(self, X):
        constraints = []

        for k in range(self.N + 1):
            constraints += [X[0, k], X[1, k]]

        return constraints

    def _make_reconstruct_trajectory(self, X, U, P):
        return casadi.Function(
            'ff',
            [casadi.vertcat(casadi.reshape(U, 2 * self.N, 1), P)],
            [X]
        )


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

    x = np.array(trajectory[0, :])[0]
    y = np.array(trajectory[1, :])[0]

    plt.plot(initial_state[0], initial_state[1], 'bx')
    plt.plot(goal_state[0], goal_state[1], 'gx')

    plt.plot(x, y)
    plt.show()
