#!/usr/bin/env python3

#import rospy

class iLQR():
    def __init__(self, dynamics, cost, N):
        """Constructs an iLQR controller.

        Args:
            dynamics: Dynamics System.
            cost: Cost function.
            N: Horizon length.
        """

        self.dynamics = dynamics
        self.cost = cost
        self.N = N

    def fit(self, x0, u0, n_iter=10, tol=1e-6):
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
        pass

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

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path.
            us: Control path.

        Returns:
            Trajectory's total cost.
        """
        pass

    def _forward_pass(self, x0, us):
        """Apply the forward dynamics

        Args:
            x0: Initial state.
            us: Control path.

        Returns:
            Tuple of:
                xs: State path.
                Q_x: Jacobian of state path w.r.t. x.
                Q_u: Jacobian of state path w.r.t. u.
                l: Cost path .
                L_x: Jacobian of cost path w.r.t. x.
                L_u: Jacobian of cost path w.r.t. u. 
                l_xx: Hessian of cost path w.r.t. x, x.
                l_ux: Hessian of cost path w.r.t. u, x.
                l_uu: Hessian of cost path w.r.t. u, u.
                Q_xx: Hessian of state path w.r.t. x, x .
                Q_ux: Hessian of state path w.r.t. u, x.
                Q_uu: Hessian of state path w.r.t. u, u.
        """
        pass

    def _backward_pass(self,
                       Q_x,
                       Q_u,
                       l_x,
                       l_u,
                       l_xx,
                       l_ux,
                       l_uu,
                       Q_xx,
                       Q_ux,
                       Q_uu):
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
        pass

if __name__ == "__main__":
    controller = iLQR()