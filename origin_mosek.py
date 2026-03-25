import mosek as msk
import numpy as np
import sys

from util import run_method
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def stream_printer(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def gl_mosek(_x0: NDArray,
             A: NDArray,
             b: NDArray,
             mu: float,
             opts: Optional[Dict] = {}) -> Tuple[NDArray, int, Dict]:
    m, n = A.shape
    _, l = b.shape

    var_cnt = 2 + n + n * l

    # Create the MOSEK environment and task
    with msk.Env() as env:
        with env.Task() as task:
            if 'log' in opts and opts['log']:
                task.set_Stream(msk.streamtype.log, stream_printer)

            if 'maxtime' in opts and opts['maxtime']>0:
                task.putdouparam(msk.dparam.optimizer_max_time, opts['maxtime'])

            task.appendvars(var_cnt)

            #设置变量
            task.putvarbound(0, msk.boundkey.lo, 0.0, np.inf)  # t >= 0
            task.putvarbound(1, msk.boundkey.fx, 0.5, 0.5)  # t0 = 0.5
            for i in range(n):
                task.putvarbound(2 + i, msk.boundkey.lo, 0.0, np.inf)  # s >= 0
            for i in range(n * l):
                task.putvarbound(2 + n + i, msk.boundkey.fr, -np.inf, np.inf)  # x(i, j) free

            #定义目标函数
            task.putcj(0, 0.5)  # 为了适应旋转二次，0.5 * t
            for i in range(n):
                task.putcj(2 + i, mu)

            task.appendafes(m * l + 2 + n * (l + 1))
            task.putafefentrylist([0], [0], [1.0])
            task.putafefentrylist([1], [1], [1.0])
            for i in range(m):
                for j in range(l):
                    col_idx = [2 + n + k * l + j for k in range(n)]
                    row_vals = [A[i, k] for k in range(n)]
                    task.putafefentrylist([2 + i * l + j] * n, col_idx, row_vals)
            task.putafegslice(2, m * l + 2, [-b[i, j] for i in range(m) for j in range(l)])
            offset = 2 + m * l
            for i in range(n):
                row_idx = [offset + i * (l + 1) + j for j in range(l + 1)]
                col_idx = [i + 2] + [2 + n + i * l + j for j in range(l)]
                row_vals = [1.0] * (l + 1)
                task.putafefentrylist(row_idx, col_idx, row_vals)

            task.appendcons(1 + n)


            r_quad_cone = task.appendrquadraticconedomain(2 + m * l)
            task.appendacc(r_quad_cone, range(m * l + 2), None)


            for i in range(n):
                quad_cone = task.appendquadraticconedomain(1 + l)
                task.appendacc(quad_cone, [2 + m * l + i * (l + 1) + j for j in range(l + 1)], None)


            task.putobjsense(msk.objsense.minimize)
            task.optimize()


            result = task.getxxslice(msk.soltype.itr, 2 + n, 2 + n + n * l)
            result = np.array(result).reshape((n, l))


            num_it = task.getintinf(msk.iinfitem.intpnt_iter)

            fval = 0.5 * np.linalg.norm(A @ result - b, 'fro') ** 2 + mu * np.sum(np.linalg.norm(result.reshape(n, -1), axis=1))

            res = {'fval':fval,'status': task.getprosta(msk.soltype.itr), 'obj': task.getprimalobj(msk.soltype.itr)}

            return result, num_it, res


if __name__ == "__main__":
    run_method(gl_mosek, plot=False)
