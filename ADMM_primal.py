from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from selenium.webdriver.support.expected_conditions import element_selection_state_to_be

from util import group_lasso_loss, extract_config, run_method


def gl_ADMM_primal(x0: NDArray,
                   A: NDArray,
                   b: NDArray,
                   mu: float,
                   opts: Optional[Dict] = {}) -> Tuple[NDArray, int, Dict]:
    r"""
    Returns
        The solver information contains the following keys:
        - 'fval': the value of the objective function at the last iterate.
        - obj_val: the list of objective values during the iterations.
    """
    m, n = A.shape
    _, l = b.shape
    iter_count = 0

    tol = extract_config(opts, 'tol', 1e-7)
    max_iter = extract_config(opts, 'maxiter', 5000)
    log = extract_config(opts, 'log', True)
    benchmark = extract_config(opts, 'benchmark', False)
    alpha = extract_config(opts, 'alpha', 1 / 5)

    x = x0
    y = np.zeros((n, l), dtype=float)
    z = np.zeros((n, l), dtype=float)

    obj_val_list = []
    grad_norm_list = []
    dual_gap_list = []

    rho = 5.0

    inv = np.linalg.inv(A.T @ A + rho * np.eye(n))
    if "step_size" in opts and opts["step_size"]>0:
        step_size = opts["step_size"]
    else:
        step_size = (1 + np.sqrt(5)) / 2

    temp = A.T @ b

    while iter_count <= max_iter:
        if log and iter_count % 50 == 0:
            print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, x, mu)}")

        iter_count += 1

        # 更新x
        x = inv @ (temp + rho * z - y)

        # 更新辅助变量
        z = x + alpha * y
        norms = np.linalg.norm(z, ord=2, axis=1, keepdims=True)
        z -= z * np.minimum(1, (mu * alpha) / norms)

        # 更新乘子
        y = y + step_size * rho * (x - z)

        if not benchmark:
            obj_val = group_lasso_loss(A, b, x, mu)
            obj_val_list.append(obj_val)
        if np.linalg.norm(x - z, ord="fro") < tol:
            break
    result = x
    fval = 0.5 * np.linalg.norm(A @ result - b, 'fro') ** 2 + mu * np.sum(np.linalg.norm(result.reshape(n, -1), axis=1))

    return x, iter_count, {'fval':fval, 'obj_val': obj_val_list, 'dual_gap': dual_gap_list}


if __name__ == '__main__':
    run_method(gl_ADMM_primal, log_scale=False, benchmark=False)
