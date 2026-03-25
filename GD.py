from typing import Tuple, Dict, Optional

import numpy as np
from numpy.ma.core import inner
from numpy.typing import NDArray

from util import group_lasso_loss, extract_config, run_method


def gl_GD_primal(x0: NDArray,
                 A: NDArray,
                 b: NDArray,
                 mu: float,
                 opts: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    r"""

    Returns :
    -------
        - 'fval': the value of the objective function at the last iterate.
        - 'obj_val': the list of objective values during the iterations.
        - 'grad_norm': the list of gradient norms during the iterations.
    """
    iter_count = 0
    x = x0
    #获取学习率
    if "step_size" in opts and opts["step_size"]>0:
        step_size = opts["step_size"]
    else:
        step_size = 1 / np.linalg.norm(A, ord=2) ** 2
    f_last = np.inf
    #提取超参数
    tol = extract_config(opts, 'tol', 1e-8)
    max_iter = extract_config(opts, 'max_iter', 5000)
    inner_max_iter=extract_config(opts, 'inner_max_iter', 100)
    log = extract_config(opts, 'log', True)
    sigma = extract_config(opts, 'sigma', 1e-6)
    benchmark = extract_config(opts, 'benchmark', False)
    clear_edge=extract_config(opts, 'clear_edge', 1e-4)
    #配置连续化策略参数
    orig_mu = mu
    mu_list = list(reversed([(4 ** i) * mu for i in range(7)]))
    sigma_list = list(reversed([sigma * (4 ** i) for i in range(7)]))
    last_idx = len(mu_list) - 1

    if log:
        print(f"Start the gradient descent algorithm with step size {step_size}.")

    obj_val_list = []
    grad_norm_list = []

    for k, (mu, sigma) in enumerate(zip(mu_list, sigma_list)):
        if iter_count > max_iter:
            break

        inner_iter = 0
        while True:
            # 输出运行日志
            if log and iter_count % 100 == 0:
                print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, x, orig_mu)}")

            iter_count += 1
            inner_iter += 1

            if iter_count > max_iter or (inner_iter > inner_max_iter and k != last_idx):
                break
            grad = A.T @ (A @ x - b)
            norms = np.linalg.norm(x, ord=2, axis=1)
            idx = norms > sigma
            grad[idx, :] += mu * x[idx, :] / norms[idx][:, None]
            grad[~idx, :] += mu * x[~idx, :] / sigma

            if k != last_idx:
                step = step_size
            else:
                step = step_size if inner_iter <= 80 else step_size / np.sqrt(inner_iter - 80)

            x -= step * grad
            x[np.abs(x) < clear_edge] = 0
            f_new = group_lasso_loss(A, b, x, mu)

            if not benchmark:
                grad_norm = np.linalg.norm(grad, ord="fro")
                obj_val_list.append(f_new)
                grad_norm_list.append(grad_norm)
            #判断终止条件
            if abs(f_last - f_new) < tol:
                break

            f_last = f_new

    return x, iter_count, {'fval':f_last,'obj_val': obj_val_list, 'grad_norm': grad_norm_list}


if __name__ == '__main__':
    run_method(gl_GD_primal, benchmark=True)
