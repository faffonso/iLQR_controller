#!/usr/bin/env python3

#import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import casadi as ca
import casadi.tools as cat
PI = 3.14159265359

class iLQR():
    def __init__(self, dynamics, cost, dt=0.1, N=18):
        """Constructs an iLQR controller.

        Args:
            dynamics: Dynamics System.
            cost: Cost function.
            dt: Sampling time
            N: Horizon length.
        """
        self.dt = dt
        self.dynamics = dynamics
        self.cost = cost

        self.N = N
        self.alpha = 1.0

        self._k = np.zeros((N, 3))
        self._K = np.zeros((N, 2, 3))


    def fit(self, x0, u0, n_iter=15, tol=1e-6):

        alphas = 0.5**np.arange(8)

        """Compute optimal control

        Args:
            x0: Initial state.
            u0: Initial control path.
            n_iter: Max iterations.
            tol: Tolerance.

        Returns:
            xs: Optimal state path.
            us: Optimal control path.
        """

        xs = x0
        us = u0  
        J_old = self._rollout(xs[0], us)

        for iteration in range(n_iter):
            # print(f'Iteration {iteration}')
            k, K, _ = self._backward_pass(xs, us)

            for alpha in alphas:
                xs_new, us_new = self._forward_pass(xs, us, k, K, alpha)
                J = self.cost._trajectory_cost(xs_new, us_new)

                print(xs_new)

                print(f'Trying convergence {J} on {J_old}')
                
                if (abs(J) < abs(J_old)):
                    J_old = J
                    xs = xs_new
                    us = us_new
                    break

            #print(xs)

            x = xs[:, 0]
            y = xs[:, 1]
            orientacao = xs[:, 2]
                
            # Plote os dados
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, c=orientacao, cmap='viridis', marker='o')
            plt.colorbar(label='Orientação (graus)')
            plt.xlabel('Coordenada X')
            plt.ylabel('Coordenada Y')
            plt.title('Plot de Posições x, y e Orientação')
            plt.grid(True)
            plt.show()




    def _control(self, xs, us, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path.
            us: Nominal control path.
            k: Feedforward gains.
            K: Feedback gains.
            alpha: Line search coefficient.

        Returns:
            xs: state path.
            us: control path.
        """
        pass

    def _forward_pass(self, xs, us, ks, Ks, alpha, hessian=False):
        """Apply the forward dynamics

        Args:
            x0: Initial state.
            us: Control path.

        Returns:
            Tuple of:
                x: State path.
                us: Action control
        """
        
        u_new = np.empty((self.N, 2))
        x_new = np.empty((self.N, 3))

        x_new = xs

        for i in range (self.N-1):
            u_new[i] = us[i] + Ks[i] @ (x_new[i] - xs[i]) + alpha * ks[i]

            x_next = xs[i] + self.dynamics.f(xs[i], us[i]) * self.dt
            
            for j in range(3):
                x_new[i+1][j] = x_next[j]

        return x_new, u_new



    def _backward_pass(self, xs, us):
        """Computes the feedforward and feedback gains k and K.

        Args:
            Q_x: Jacobian of state path w.r.t. x.
            Q_u: Jacobian of state path w.r.t. u.
            l_x: Jacobian of cost path w.r.t. x.
            l_u: Jacobian of cost path w.r.t. u.
            l_xx: Hessian of cost path w.r.t. x, x.
            l_ux: Hessian of cost path w.r.t. u, x.
            l_uu: Hessian of cost path w.r.t. u, u.
            Q_xx: Hessian of state path w.r.t. x, x.
            Q_ux: Hessian of state path w.r.t. u, x.
            Q_uu: Hessian of state path w.r.t. u, u

        Returns:
            k: feedforward gains.
            K: feedback gains.
        """
        
        ks = np.empty(us.shape)
        Ks = np.empty((us.shape[0], us.shape[1], xs.shape[1]))

        delta_V = 0

        Qf = cost.get_Qf()

        v_x = Qf @ xs[0]
        v_xx = Qf
        for n in range(us.shape[0] - 1):

            f_x, f_u = self.dynamics.f_prime(xs[n], us[n])
            l_x, l_u, l_xx, l_ux, l_uu  = self.cost.l_prime(xs[n], us[n])
        

            # Q_terms
            Q_x  = l_x  + f_x.T @ v_x
            Q_u  = l_u  + f_u.T @ v_x
            Q_xx = l_xx + f_x.T @ v_xx @ f_x
            Q_uu = l_uu + f_u.T @ v_xx @ f_u
            Q_ux = l_ux + f_u.T @ v_xx @ f_x

            k = -np.linalg.inv(Q_uu) @ Q_u
            K = -np.linalg.inv(Q_uu) @ Q_ux
            
            k = np.array(k)
            K = np.array(K)

            ks[n], Ks[n] = k.T, K

            # V_terms
            v_x  = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
            v_xx = Q_xx + K.T@Q_ux + Q_ux.T @ K + K.T @ Q_uu @ K
            #expected cost reduction
            delta_V += Q_u.T@k + 0.5*k.T@Q_uu@k

        return ks, Ks, delta_V
        
    def _rollout(self, x0, us):
        J = 0
        N = us.shape[0]

        xs = np.empty((N, 3))
        xs[0] = x0


        for n in range(N - 1):
            x_next = self.dynamics.f(xs[n], us[n]) * self.dt
            for j in range(3):
                xs[n+1][j] = xs[n][j] + x_next[j]
            J += self.cost._l(xs[n], us[n])[0]
        
        return J
    
class Dynamics():
    def __init__(self, vr, omegar, dt=.1):

        self.dt = dt
        self.vr = vr
        self.omegar = omegar

        x = ca.SX.sym('e', 3)
        u = ca.SX.sym('u', 2)

        x1_dot = vr * ca.cos(x[2]) + u[1] * x[1] - u[0]
        x2_dot = vr * ca.sin(x[2]) + u[1] * x[0]
        x3_dot = omegar - u[1]

        x_dot = ca.veccat(x1_dot, x2_dot, x3_dot)

        
        self._f = ca.Function('f', [x, u], [x_dot])
        self._A = ca.Function('A', [x, u], [ca.jacobian(self._f(x, u), x)])
        self._B = ca.Function('B', [x, u], [ca.jacobian(self._f(x, u), u)])
        

    def f(self, x, u):
        return self._f(x, u)

    def f_prime(self, x, u):
        f_x = self._A(x, u)
        f_u = self._B(x, u)
        return f_x, f_u

class Cost():
    def __init__(self, Q, R, Qf):
        x = ca.SX.sym('e', 3)
        u = ca.SX.sym('u', 2)

        self.Qf = Qf
        self.Q = Q
        self.R = R

        cost = 1/2 * (x.T @ Q @ x + u.T @ R)

        self._l = ca.Function('l', [x, u], [cost])
        self._lx = ca.Function('lx', [x, u], [ca.jacobian(self._l(x, u), x)])
        self._lu = ca.Function('lu', [x, u], [ca.jacobian(self._l(x, u), u)])
        
    
    def _trajectory_cost(self, xs, us):
        J = 0
        for x, u in zip(xs, us):
            J += self._l(x, u)[0]

        return J

    def l_prime(self, x, u):
        l_x = self.Q @ x
        l_u = self.R @ u

        l_xx = self.Q
        l_uu = self.R
        l_xu = 0

        return l_x, l_u, l_xx, l_xu, l_uu
    
    def get_Qf(self):
        return self.Qf
    
if __name__ == "__main__":
    vr = .1
    omegar = 0.0
    N = 10

    Q = np.diag([1.0, 1.0, 1.0])
    R = np.diag([1.0, 1.0])
    Qf = np.diag([1.0, 1.0, 1.0])

    dynamics = Dynamics(vr, omegar)
    cost = Cost(Q, R, Qf)

    x0 = np.array([[-2.0, -1.0, 0]] * N, dtype=float)
    u0 = np.array([[vr, 0.0]] * N)

    print(u0)

    controller = iLQR(dynamics, cost, N=N)
    controller.fit(x0, u0)

