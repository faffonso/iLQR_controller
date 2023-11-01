from casadi import *

if __name__ == "__main__":
    import casadi as ca

    # Definir variáveis de estado e controle
    x = SX.sym('e', 3)
    u = SX.sym('u', 2)

    x1_dot = u[0] * sin(x[2])
    x2_dot = u[0] * cos(x[2])
    x3_dot = u[1]

    xdot = veccat(x1_dot, x2_dot, x3_dot)

    f = Function('f', [x, u], [xdot])

    f_x = ca.Function('Af', [x, u], [ca.jacobian(f(x, u), x)])
    f_u = ca.Function('Bf', [x, u], [ca.jacobian(f(x, u), u)])

    f_xx = ca.Function('f_xx', [x, u], [jacobian(jacobian(f(x, u), x), x)])
    f_uu = ca.Function('f_xx', [x, u], [jacobian(jacobian(f(x, u), u), u)])
    f_xu = ca.Function('f_xx', [x, u], [jacobian(jacobian(f(x, u), x), u)])

    print(f_x([5, 2, 0], [1, 0]))
    print(f_u([5, 2, 0], [1, 0]))

    print(f_xx([5, 2, 0], [1, 0]))
    print(f_uu([5, 2, 0], [1, 0]))   
    print(f_xu([5, 2, 0], [1, 0]))