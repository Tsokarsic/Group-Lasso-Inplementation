from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from util import group_lasso_loss, extract_config, run_method


def gl_FProxGD_primal(x0: NDArray,
                      A: NDArray,
                      b: NDArray,
                      mu: float,
                      opts: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    """

    Returns
    -------
        - 'fval': the value of the objective function at the last iterate.
        - 'obj_val': the list of objective values during the iterations.
        - 'grad_norm': the list of gradient norms during the iterations.
    """
    _, l = b.shape
    iter_count = 0
    x = x0
    #初始学习率
    if "step_size" in opts and opts["step_size"]>0:
        step_size = opts["step_size"]
    else:
        step_size = 1 / np.linalg.norm(A, ord=2) ** 2
    f_last = np.inf
    tol = extract_config(opts, 'tol', 1e-8)
    max_iter = extract_config(opts, 'maxiter', 5000)
    log = extract_config(opts, 'log', True)
    benchmark = extract_config(opts, 'benchmark', False)
    orig_mu = mu
    #连续化策略
    mu_list = list(reversed([(6 ** i) * mu for i in range(5)]))
    last_idx = len(mu_list) - 1

    obj_val_list = []
    grad_norm_list = []
    last_x = [x.copy(), x.copy()]
    #外循环
    for k, mu in enumerate(mu_list):
        if iter_count > max_iter:
            break

        inner_iter = 0
        #内循环
        while True:
            if log and iter_count % 20 == 0:
                print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, x, orig_mu)}")

            iter_count += 1
            inner_iter += 1

            if iter_count > max_iter or (inner_iter > 50 and k != last_idx):
                break

            if k != last_idx:
                step = step_size
            else:
                step = step_size if inner_iter <= 100 else step_size / np.sqrt(inner_iter - 100)

            y = last_x[1] + (inner_iter - 2) / (inner_iter + 1) * (last_x[1] - last_x[0])
            grad = A.T @ (A @ y - b)
            x = y - step * grad
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x -= x * np.minimum(1, mu * step / norms)
            x[np.abs(x) < 1e-4] = 0

            f_new = group_lasso_loss(A, b, x, mu)

            if not benchmark:
                grad_norm = np.linalg.norm(grad, ord="fro")
                obj_val_list.append(f_new)
                grad_norm_list.append(grad_norm)

            if abs(f_last - f_new) < tol:
                break

            f_last = f_new
            last_x[0] = last_x[1]
            last_x[1] = x.copy()

    return x, iter_count, {'fval':f_last, 'obj_val': obj_val_list, 'grad_norm': grad_norm_list}


if __name__ == '__main__':
    run_method(gl_FProxGD_primal, benchmark=True)
