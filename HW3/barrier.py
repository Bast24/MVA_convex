from utils import generate_data, build_lasso_params
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler


""" Parameters """
# Matplotlib parameters
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)


MAX_ITER = 100

# For backtracking line search
ALPHA = 0.25
BETA = 0.9

# For barrier method
T0 = 1
MU = 10


def dual_objective(Q, p, v):
    """Return the objective value for the dual problem of Lasso.

    Args:
        - Q, p : parameters of QP
        - v : evaluation point

    Returns:
        Value of the objective function
    """
    return np.vdot(Q @ v + p, v)


def f_0_dual(Q, p, v):
    return v.T @ Q @ v + p.T @ v


def f_0_primal(X, Y, w):
    return 0.5 * np.linalg.norm(X @ w - Y) ** 2 + lmbda * np.linalg.norm(w, 1)


def dual_to_primal(X, Y, v):
    """Compute the primal solution given the dual one
    """
    return np.linalg.inv(X.T @ X) @ X.T @ (v + Y)


def barrier_oracle(Q, p, A, b, v, t, which=[0, 1, 1]):
    """Oracle computing the gradient and the hessian of the objective function for the barier method

    Args:
        - Q, p, A, b : parameters of QP
        - t : barrier method parameter
        - v : evaluation point
        - which : list of 3 booleans informing what to compute : the value of the function, its gradient, and/or its hessian
    
    Returns:
        - L : List of values/arrays computed depending on `which`
    """
    L = []
    d = b - A @ v  # Avoid to recompute it all the time

    if which[0]:
        # Check if the point is feasible
        if not (d > 0).all():
            raise ValueError('v is not feasible !')
        # Compute the value of the barrier objective
        val = t * f_0_dual(Q, p, v) - np.sum(np.log(d))
        L.append(val)
    
    if which[1]:
        # Compute its gradient 
        grad_f0 =  2 * Q @ v + p
        grad_phi = A.T @ (1 / d)
        grad = t * grad_f0 + grad_phi
        L.append(grad)

    if which[2]:
        # Compute its hessian
        hess_phi = A.T @ np.diag(1. / d ** 2) @ A
        hess = 2 * t * Q + hess_phi
        L.append(hess)
    
    return L

def barrier_obj(Q, p, A, b, v, t):
    return barrier_oracle(Q, p, A, b, v, t, which=[1, 0, 0])[0]


def backtracking(Q, p, A, b, t, v, dv_n, lambda_sq, alpha=ALPHA, beta=BETA):
    """Backtracking line search

    Args:
        - Q, p, A, b : parameters of QP
        - t : barrier method parameter
        - dv_n : newton direction
        - lambda_n : newton decrement
        - alpha, beta : backtracking parameters

    Returns:
        - step_size : backtracking step size
    """
    step_size = 1.
    for _ in range(MAX_ITER):
        next_point_feasible = (b - A @ (v + step_size * dv_n) > 0).all()
        
        # Stopping criterion
        if not next_point_feasible:
            break
        if barrier_obj(Q, p, A, b, v + step_size * dv_n, t) > barrier_obj(Q, p, A, b, v, t) - alpha * step_size * lambda_sq:
            break

        step_size *= beta
        
    return step_size


def centering_step(Q, p, A, b, t, v0, eps):
    """Newton method solving the centering step of the barrier method

    Args:
        - Q, p, A, b : parameters of QP
        - t : barrier method parameter
        - v0 : initial variable
        - eps : target precision

    Returns:
        - V_center : sequence of (v_i) over the iterations
        - n_iter : number of iterations to obtain eps precision
    """
    V_center = [v0]
    n_iter = 0

    for _ in range(MAX_ITER):
        grad, hess = barrier_oracle(Q, p, A, b, V_center[-1], t, [0, 1, 1])
        hess_inv = np.linalg.inv(hess)
      
        # Compute Newton step and decrement
        dv_n = -hess_inv @ grad
        lambda_sq = -np.vdot(dv_n, grad)
        
        # Stopping criterion
        if lambda_sq / 2. <= eps:
            break

        # Line search by backtracking
        step_size = backtracking(Q, p, A, b, t, V_center[-1], dv_n, lambda_sq)

        # Update
        V_center.append(V_center[-1] + step_size * dv_n)
        n_iter += 1
    
    return V_center, n_iter


def barr_method(Q, p, A, b, v0, eps, t_0=T0, mu=MU):
    """Barrier method to solve QP problem

    Args:
        - Q, p, A, b : parameters of the QP
        - t : barrier method parameter
        - v0 : feasible initial point
        - eps : target precision

    Returns:
        - V : sequence of (v_i) over the iterations
        - n_iters : number of iterations to obtain eps precision for each newton step
    """
    t = t_0
    m = len(b)
    V = [v0]
    n_iters = [0]

    for _ in range(MAX_ITER):
        # Centering step
        V_center, n_iter = centering_step(Q, p, A, b, t, V[-1], eps)

        # Stopping criterion
        if m / t <= eps:
            break
        
        # Updating
        t *= mu
        V.append(V_center[-1])
        n_iters.append(n_iters[-1] + n_iter)
    
    return V, n_iters


if __name__ == '__main__':
    # Parameters
    n = 1000
    d = 20
    lmbda = 10.

    # Generate some data
    X, Y, w = generate_data(n, d)

    # Build the associated QP parameters for LASSO
    Q, p, A, b = build_lasso_params(X, Y, lmbda)

    # Barrier method
    eps = 1e-7
    mus = [2, 15, 50, 100, 200]
    v0 = np.zeros(n)

    plt.figure()

    for mu in mus:
        V, n_it = barr_method(Q, p, A, b, v0, eps, mu=mu)
        v_star = V[-1]
        values = [dual_objective(Q, p, v) - dual_objective(Q, p, v_star) for v in V]
        plt.step(n_it, values, label='mu = {}'.format(mu))
    
    plt.semilogy()
    plt.xlabel('Newton iterations')
    plt.ylabel('$f(v_t) - f^*$')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()

    # Plot the dual gap evolution
    for mu in mus:
        V, n_it = barr_method(Q, p, A, b, v0, eps, mu=mu)
        W = [dual_to_primal(X, Y, v) for v in V]
        values = [f_0_primal(X, Y, W[i]) + f_0_dual(Q, p, V[i]) for i in range(len(V))]
        plt.step(n_it, values, label='mu = {}'.format(mu))

    plt.semilogy()
    plt.xlabel('Newton iterations')
    plt.ylabel('duality gap')
    plt.legend()
    plt.legend(loc='upper right')
    plt.show()
