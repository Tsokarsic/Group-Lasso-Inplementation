from typing import Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from util import group_lasso_loss, extract_config, run_method


def gl_ALM_dual(x0: NDArray,
                A: NDArray,
                b: NDArray,
                mu: float,
                opts: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    r"""
    Returns

        The solver information contains the following keys:
        - 'fval': the value of the objective function at the last iterate.
        - obj_val: the list of objective values during the iterations.
        - dual_gap: the list of dual gaps during the iterations.
    """
    m, n = A.shape
    _, l = b.shape
    iter_count = 0

    tol = extract_config(opts, 'tol', 1e-8)
    tol_inner = extract_config(opts, 'tol_inner', 1e-4)
    max_iter = extract_config(opts, 'maxiter', 5000)
    log = extract_config(opts, 'log', True)
    benchmark = extract_config(opts, 'benchmark', False)

    x = x0
    y = np.zeros((m, l), dtype=float)
    z = np.zeros((n, l), dtype=float)

    obj_val_list = []
    dual_gap_list = []

    sigma = 100.0
    total_iter = 0

    inv = np.linalg.inv(sigma * A @ A.T + np.eye(m))
    if "step_size" in opts and opts["step_size"]>0:
        step_size = opts["step_size"]
    else:
        step_size = (1 + np.sqrt(5)) / 2
    while iter_count <= max_iter:
        if log:
            print(f"Iteration: {iter_count}, Objective value: {group_lasso_loss(A, b, -x, mu)}")

        iter_count += 1

        for i in range(50):
            total_iter += 1
            # 步骤1：闭式更新Y (固定X、Z)
            y = inv @ (A @ (sigma * z - x) - b)
            # 步骤2：无约束更新Z_raw
            z_raw = x / sigma + A.T @ y
            # 步骤3：投影到||Z||_{∞,2} ≤ μ约束集
            row_norms = np.linalg.norm(z_raw, axis=1, keepdims=True)
            row_norms_clipped = np.maximum(row_norms, mu)
            z_new = z_raw * (mu / row_norms_clipped)
            # 收敛判断
            if np.linalg.norm(z_new - z, ord="fro") < tol_inner:
                z = z_new
                break
            z = z_new

        x = x + step_size * sigma * (A.T @ y - z)

        if not benchmark:
            obj_val = group_lasso_loss(A, b, -x, mu)
            obj_val_list.append(obj_val)
            dual_val = np.sum(b * y) + 0.5 * np.linalg.norm(y, "fro") ** 2
            dual_gap_list.append(abs(obj_val - dual_val))
        if np.linalg.norm(A.T @ y - z, ord="fro") < tol:
            break

    if log:
        print(f"Total inner iterations: {total_iter}")
    result=-x
    fval = 0.5 * np.linalg.norm(A @ result - b, 'fro') ** 2 + mu * np.sum(np.linalg.norm(result.reshape(n, -1), axis=1))

    return -x, iter_count, {'fval':fval,'obj_val': obj_val_list, 'dual_gap': dual_gap_list}


if __name__ == '__main__':
    run_method(gl_ALM_dual, benchmark=True)
