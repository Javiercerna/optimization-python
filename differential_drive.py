'''
Adapted from this repository:
https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi/tree/master/workshop_github
'''

import casadi
import numpy as np

import math

N = 80  # prediction horizon
dt = 0.2  # sampling time

speed_limits = [-0.6, 0.6]
angular_speed_limits = [-math.pi / 4, math.pi / 4]
position_limits = [-10, 10]  # Assume symmetric for x and y


x = casadi.SX.sym('x')
y = casadi.SX.sym('y')
angle = casadi.SX.sym('angle')
speed = casadi.SX.sym('speed')
angular_speed = casadi.SX.sym('angular_speed')

states = [x, y, angle]
controls = [speed, angular_speed]

nx = len(states)
nu = len(controls)

control_action = [
    dt * speed * casadi.cos(angle),
    dt * speed * casadi.sin(angle),
    dt * angular_speed
]

f = casadi.Function(
    'f', states + controls, control_action,
    ['x', 'y', 'angle', 'speed', 'angular_speed'], ['d_x', 'd_y', 'd_angle'])

U = casadi.SX.sym('U', nu, N)
P = casadi.SX.sym('P', 2 * nx)  # Initial state + reference state

X = casadi.SX.sym('X', nx, N + 1)

X[:, 0] = P[0:nx]

for k in range(N):
    f_values = f(
        x=X[:, k][0], y=X[:, k][1], angle=X[:, k][2],
        speed=U[:, k][0], angular_speed=U[:, k][1]
    )
    X[:, k + 1] = X[:, k] + casadi.vertcat(*f_values.values())

ff = casadi.Function(
    'ff',
    [casadi.vertcat(casadi.reshape(U, 2 * N, 1), P)],
    [X]
)

J = 0
constraints = []

Q = casadi.SX([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

R = casadi.SX([[1, 0],
               [0, 1]])

for k in range(N):
    state_cost = (X[:, k] - P[nx:]).T @ Q @ (X[:, k] - P[nx:])
    input_cost = U[:, k].T @ R @ U[:, k]
    J += state_cost + input_cost

for k in range(N + 1):
    constraints += [X[0, k], X[1, k]]


nlp = {
    'x': casadi.reshape(U, 2 * N, 1),
    'f': J,
    'g': casadi.vertcat(*constraints),
    'p': P
}

S = casadi.nlpsol('S', 'ipopt', nlp)

lower_bound_x = [speed_limits[0], angular_speed_limits[0]] * N
upper_bound_x = [speed_limits[1], angular_speed_limits[1]] * N

initial_state = [0, 0, 0]
goal_state = [5, 5, 0]
initial_inputs = np.zeros((N, 2))

result = S(
    x0=casadi.reshape(initial_inputs.T, 2 * N, 1),
    lbg=position_limits[0],
    ubg=position_limits[1],
    lbx=lower_bound_x,
    ubx=upper_bound_x,
    p=initial_state + goal_state
)

trajectory = ff(casadi.vertcat(result['x'], initial_state, goal_state))

print(trajectory)
