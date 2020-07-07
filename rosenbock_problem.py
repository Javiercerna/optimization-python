'''
Taken from here: https://web.casadi.org/docs/#nonlinear-root-finding-problems
'''

import casadi

x = casadi.SX.sym('x')
y = casadi.SX.sym('y')
z = casadi.SX.sym('z')

nlp = {
    'x': casadi.vertcat(x, y, z),
    'f': x ** 2 + 100 * z ** 2,
    'g': z + (1 - x) ** 2 - y
}

S = casadi.nlpsol('S', 'ipopt', nlp)

result = S(x0=[2.5, 3.0, 0.75], lbg=0, ubg=0)

print('Optimal solution: {}'.format(result["x"]))
