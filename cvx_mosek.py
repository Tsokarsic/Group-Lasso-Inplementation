import cvxpy as cp

from util import run_method
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def gl_cvx_mosek(x0: NDArray,
                 A: NDArray,
                 b: NDArray,
                 mu: float,
                 opts: Optional[Dict] = None) -> Tuple[NDArray, int, Dict]:
    m, n = A.shape
    _, l = b.shape
    x = cp.Variable((n, l))
    x.value = x0
    obj = 0.5 * cp.norm(A @ x - b, "fro") ** 2 + mu * cp.sum(cp.norm(x, axis=1))
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.MOSEK)
    out = {
        "fval": prob.value,  # 目标函数最优值（核心）
        "status": prob.status,  # 求解状态（如"optimal"）
        "solver_stats": prob.solver_stats  # 保留原始SolverStats对象
    }

    # 4. 提取迭代次数和最优解
    iter1 = prob.solver_stats.num_iters  # 从SolverStats取迭代次数
    x1 = x.value  # 提取最优解

    # 5. 返回格式保持一致：x, iter, out
    return x1, iter1, out


if __name__ == "__main__":
    run_method(gl_cvx_mosek, plot=False)
